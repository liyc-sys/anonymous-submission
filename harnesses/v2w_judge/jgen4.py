#!/usr/bin/env python3
"""JudgeHarness jgen4 — jgen2 contract + hi-res tiles + DOM text fidelity.

Task: given an unlabeled pair (task name + two candidate render directories
A/B), decide which candidate is visually closer to the prototype screenshots.
Writes verdicts.jsonl.

Evolution hard constraints (same as jgen2):
1. Only the single model named by --model may be called;
2. No official score/answer files may be read at runtime;
3. Zero task-specific content in the code;
4. At most 3 LLM calls per pair judgement (here exactly 3: one per viewport).

Changes over jgen2 (targeting the "thumbnail strips make text/details
unreadable" diagnosis):
- Per viewport, on top of jgen2's global small strip, append hi-res vertical
  tiles of PROTOTYPE/X/Y each
  (original render width, each tile <=1150px tall, at most 3 tiles per side),
  with a single chat.completions call
  carrying multiple images (total call count unchanged, still exactly one per viewport).
- The document.body.innerText additionally dumped by the render side is
  injected into the prompt as text evidence
  (truncated to 1500 chars per candidate); the prototype has no DOM, so its
  text must be read from the hi-res tiles.
- The prompt adds a "text/content fidelity" criterion and requires comparing
  tile-by-tile first, then judging overall.
- Abstention/tie handling and the defect-aware mechanical tiebreak are
  completely identical to jgen2 (copied verbatim below; do not change the logic).
"""
import argparse, base64, hashlib, io, json, math, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# same as jgen2: one call per viewport + global strip panel width
VIEWPORTS = (("desktop", 520), ("tablet", 400), ("mobile", 320))

# hi-res tiles: original render width (desktop 1920 / tablet 1024 / mobile 375), no scaling;
# each tile <=1150px tall, at most 3 tiles per side to bound tokens.
TILE_H = 1150
MAX_TILES = 3
TEXT_CHARS = 1500

PROMPT = """You are comparing two candidate webpages X and Y against a PROTOTYPE design, for the {vp} viewport.
You are given, in order:
1. One small overview strip laid out as [ PROTOTYPE | X | Y ] (global layout only).
2. Hi-resolution vertical tiles of the PROTOTYPE (top to bottom).
3. Hi-resolution vertical tiles of candidate X (top to bottom).
4. Hi-resolution vertical tiles of candidate Y (top to bottom).

Visible text extracted from each candidate's DOM (reading order, truncated):
[X text]: {x_text}

[Y text]: {y_text}

(The prototype has no DOM text — read its text from the hi-res PROTOTYPE tiles.)

Method: FIRST compare tile-by-tile (top slice vs top slice, middle vs middle, bottom vs bottom),
THEN make an overall decision.

Judge by, in rough priority order:
1. Overall layout, section structure and section ORDER (header, hero/content, sidebar, footer).
2. Presence and correct placement of the major blocks (nav bar, main text, images, footer columns).
3. Dominant colors / background / header & footer theme.
4. TEXT / CONTENT FIDELITY: which candidate's visible text (headings, nav labels, paragraphs,
   button labels, footer text) matches the prototype's text more closely. Use both the DOM
   text above and the hi-res tiles.

IMPORTANT — treat these as making a candidate a WORSE match, even if it otherwise looks cleaner:
- Broken images, black/placeholder image boxes, or missing images that the prototype has.
- Grossly OVERSIZED or MISPLACED images/icons (e.g. giant icons that blow up a section).
- A page that is far too TALL or too SHORT compared to the prototype (wrong vertical scale).
- Missing or truncated navigation / footer.

Do NOT reward a candidate merely for looking clean or having a pleasing color if it drops
sections, breaks images, or badly distorts the vertical proportions of the prototype.

Answer STRICT JSON only: {{"winner": "X"}} or {{"winner": "Y"}} or {{"winner": "tie"}}"""


def load_img(p, width):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    im = im.resize((width, max(1, round(im.height * width / im.width))))
    if im.height > 2200:
        im = im.crop((0, 0, im.width, 2200))
    return im


def make_strip(proto_p, left_p, right_p, panel_w, out_p):
    from PIL import Image, ImageDraw
    ims = [load_img(proto_p, panel_w), load_img(left_p, panel_w), load_img(right_p, panel_w)]
    H = max(im.height for im in ims) + 26
    canvas = Image.new("RGB", (panel_w * 3 + 24, H), (24, 24, 24))
    dr = ImageDraw.Draw(canvas)
    for i, (im, name) in enumerate(zip(ims, ("PROTOTYPE", "X", "Y"))):
        x = i * (panel_w + 12)
        dr.text((x + panel_w // 2 - 24, 5), name, fill=(255, 230, 90))
        canvas.paste(im, (x, 26))
    canvas.save(out_p, "JPEG", quality=88)
    return out_p


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def load_tiles(p):
    """Full-width top->bottom tiles of height <= TILE_H, at most MAX_TILES."""
    from PIL import Image
    im = Image.open(p).convert("RGB")
    tiles, y = [], 0
    while y < im.height and len(tiles) < MAX_TILES:
        tiles.append(im.crop((0, y, im.width, min(im.height, y + TILE_H))))
        y += TILE_H
    return tiles


def img_part(im, quality=85):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return {"type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def read_dom_text(p):
    try:
        txt = Path(p).read_text(errors="replace")
    except Exception:
        return "(unavailable)"
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt:
        return "(empty)"
    return txt[:TEXT_CHARS]


def _swap(pair_id, vp):
    """Deterministic (not PYTHONHASHSEED-dependent) left/right swap for debias."""
    h = int(hashlib.md5(f"{pair_id}|{vp}".encode()).hexdigest(), 16)
    return (h % 2) == 1


# ---------- mechanical tiebreaker (LLM-free, defect-aware) — copied verbatim from jgen2 ----------
def _feat(p):
    """Return (aspect=height/width, normalized 48x48 histogram)."""
    from PIL import Image
    im = Image.open(p).convert("RGB")
    aspect = im.height / max(1, im.width)
    hist = im.resize((48, 48)).histogram()
    s = sum(hist) or 1
    hist = [x / s for x in hist]
    return aspect, hist


def _dist(fp, fc):
    """Distance of candidate feature fc from prototype fp.

    Height (vertical scale) is the primary signal: a candidate whose rendered
    page height is far from the prototype has a gross layout defect (oversized
    images, dropped sections). Color histogram is a weak secondary tiebreaker
    (prototypes with dark photo heros make pure color unreliable).
    """
    ap, hp = fp
    ac, hc = fc
    # log height ratio -> symmetric, penalizes gross over/under-height heavily
    hgt = abs(math.log((ac + 1e-6) / (ap + 1e-6)))          # ~0..~1+
    hd = sum(abs(x - y) for x, y in zip(hp, hc))            # ~0..2
    return 2.0 * hgt + 0.4 * hd


def mech_verdict(proto_paths, a_paths, b_paths):
    try:
        da = db = 0.0
        for vp in proto_paths:
            fp = _feat(proto_paths[vp])
            da += _dist(fp, _feat(a_paths[vp]))
            db += _dist(fp, _feat(b_paths[vp]))
        if abs(da - db) < 0.03 * max(da, db, 1e-6):
            return "tie"
        return "A" if da < db else "B"
    except Exception:
        return "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--datasets", default="./datasets/webpage")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model", default="MiniMax-M3")
    ap.add_argument("--base-url", default="https://api.example.com/v1")
    ap.add_argument("--api-key-file", default="./.api_key")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    exam = json.loads(Path(args.exam).read_text())
    R, W = Path(args.renders), Path(args.workdir)
    DS = Path(args.datasets)
    W.mkdir(parents=True, exist_ok=True)
    outf = Path(args.out)

    done = set()
    if outf.exists():
        for ln in outf.read_text().splitlines():
            try:
                done.add(json.loads(ln)["id"])
            except Exception:
                pass
    todo = [e for e in exam if e["id"] not in done]

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=Path(args.api_key_file).read_text().strip(), timeout=180)
    import threading
    lock = threading.Lock()

    def ask_viewport(vp, w, e):
        """Return 'A'/'B'/None(abstain) for one viewport (exactly one LLM call)."""
        swap = _swap(e["id"], vp)
        left_cand, right_cand = ("B", "A") if swap else ("A", "B")
        left_path = R / e[left_cand] / f"{vp}.png"
        right_path = R / e[right_cand] / f"{vp}.png"
        proto_path = DS / e["task"] / "prototypes" / f"{vp}.jpg"
        # 1) global overview strip (same as jgen2, cached)
        cp = W / f'{e["id"]}_{vp}.jpg'
        if not cp.exists():
            make_strip(proto_path, left_path, right_path, w, cp)
        # 2) hi-res tiles per side (top -> bottom)
        proto_tiles = load_tiles(proto_path)
        left_tiles = load_tiles(left_path)
        right_tiles = load_tiles(right_path)
        # 3) DOM text evidence (left/right map to X/Y)
        x_text = read_dom_text(R / e[left_cand] / "dom.txt")
        y_text = read_dom_text(R / e[right_cand] / "dom.txt")
        content = [{"type": "text", "text": PROMPT.format(vp=vp, x_text=x_text, y_text=y_text)},
                   {"type": "text", "text": "[overview strip: PROTOTYPE | X | Y]"},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(cp)}}]
        for name, tiles in (("PROTOTYPE", proto_tiles), ("X", left_tiles), ("Y", right_tiles)):
            n = len(tiles)
            for i, t in enumerate(tiles):
                label = (f"[{name} hi-res tile {i + 1}/{n} (top to bottom)]"
                         if n > 1 else f"[{name} hi-res full page]")
                content.append({"type": "text", "text": label})
                content.append(img_part(t))
        r = client.chat.completions.create(model=args.model, temperature=0.0,
                                           messages=[{"role": "user", "content": content}])
        raw = r.choices[0].message.content or ""
        m = re.search(r'"winner"\s*:\s*"(X|Y|tie)"', raw)
        if not m:
            m = re.search(r'\b(X|Y|tie)\b', raw)
        if not m:
            return None
        tok = m.group(1)
        if tok == "tie":
            return None
        return left_cand if tok == "X" else right_cand

    def run(e):
        try:
            votes = []
            for vp, w in VIEWPORTS:
                try:
                    v = ask_viewport(vp, w, e)
                except Exception:
                    v = None
                if v in ("A", "B"):
                    votes.append(v)
            a = votes.count("A")
            b = votes.count("B")
            if a > b:
                winner = "A"
            elif b > a:
                winner = "B"
            else:
                # tie or all-abstain -> defect-aware mechanical last resort (same as jgen2)
                proto_paths = {vp: DS / e["task"] / "prototypes" / f"{vp}.jpg" for vp, _ in VIEWPORTS}
                a_paths = {vp: R / e["A"] / f"{vp}.png" for vp, _ in VIEWPORTS}
                b_paths = {vp: R / e["B"] / f"{vp}.png" for vp, _ in VIEWPORTS}
                winner = mech_verdict(proto_paths, a_paths, b_paths)
            # per-viewport vote detail (counted LLM votes only; abstentions excluded)
            # — consumed by margin tie-breakers downstream; readers that only use
            # {"id","winner"} are unaffected.
            rec = {"id": e["id"], "winner": winner, "votes": {"A": a, "B": b}}
        except Exception as ex:
            rec = {"id": e["id"], "winner": "tie", "error": str(ex)[:120]}
        with lock:
            with open(outf, "a") as f:
                f.write(json.dumps(rec) + "\n")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(run, todo))
    print(json.dumps({"judged": len(todo), "skipped": len(done), "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
