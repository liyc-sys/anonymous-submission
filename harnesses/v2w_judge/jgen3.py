#!/usr/bin/env python3
"""JudgeHarness jgen3 — per-viewport high-resolution independent voting + position debiasing + defect-aware mechanical tiebreak.

Task: given an unlabeled pair (task name + two candidate render directories
A/B), decide which candidate is visually closer to the prototype screenshots.
Writes verdicts.jsonl.

Evolution constraints:
1. Only the single model named by --model may be called;
2. No official score/answer files may be read at runtime;
3. Zero task-specific content in the code;
4. At most 3 LLM calls per pair (here exactly 3: one per viewport).

Changes over jgen2 (targeting near-tie misjudgments at small gaps (~0.05)):
- Observation: the motivating misjudgment was a 0.0486 near tie — the two
  candidates render nearly the same page (layout / header / blue
  multi-column footer / colours interchangeable); the only difference is a
  tiny broken icon in Y's footer versus a row of placeholder badge boxes in
  X. The judge applied the "broken images are worse" rule (written to punish
  whole broken hero/section images) to a corner icon, over-penalizing Y and
  flipping a near tie inside the noise floor.
- Fix 1 (prompt): explicitly separate "whole/hero-level broken images or
  dropped sections" (heavy penalty) from "a single small icon or small
  footer graphic flaw" (light penalty, never decisive when everything else
  is comparable). The heavy penalty for whole-block defects is unchanged —
  that is where the adversarial-hardening gains came from.
- Fix 2 (mechanical tiebreak): the broken-image-box signal only counts when
  the near-solid box area exceeds ~8% of the panel, so small footer graphics
  cannot trigger it; still used only as a tiebreak.
- Everything else (per-viewport independent voting + per-viewport position
  debiasing + defect-aware tiebreak) is unchanged.
"""
import argparse, base64, hashlib, io, json, math, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIEWPORTS = (("desktop", 520), ("tablet", 400), ("mobile", 320))

PROMPT = """You will see one horizontal strip laid out as [ PROTOTYPE | X | Y ] for a single viewport.
The PROTOTYPE is the design target. X and Y are two candidate implementations.
Decide which candidate matches the PROTOTYPE better for THIS viewport.

Judge by, in rough priority order:
1. Overall layout, section structure and section ORDER (header, hero/content, sidebar, footer).
2. Presence and correct placement of the major blocks (nav bar, main text, images, footer columns).
3. Dominant colors / background / header & footer theme.

IMPORTANT — treat these as making a candidate a WORSE match, even if it otherwise looks cleaner:
- WHOLE broken/hero images, large black/placeholder image boxes, or missing MAJOR images the prototype has.
- Grossly OVERSIZED or MISPLACED images/icons (e.g. giant icons that blow up a section).
- A page that is far too TALL or too SHORT compared to the prototype (wrong vertical scale).
- Missing or truncated navigation / footer sections.

CALIBRATION — do not over-penalize tiny defects:
- A SINGLE small broken icon or a small footer/badge glitch is MINOR. It should NOT decide the
  winner when the two candidates are otherwise near-identical in layout, sections and color.
- Weigh dropped sections, wrong section order, whole broken hero images, and gross vertical
  distortion FAR more heavily than a lone small icon.
- If X and Y match the prototype about equally on all major structure and color, answer "tie"
  rather than flipping the decision on a minor corner artifact.

Do NOT reward a candidate merely for looking clean or having a pleasing color if it drops
sections, breaks major images, or badly distorts the vertical proportions of the prototype.

Answer STRICT JSON only: {"winner": "X"} or {"winner": "Y"} or {"winner": "tie"}"""


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


def _swap(pair_id, vp):
    """Deterministic (not PYTHONHASHSEED-dependent) left/right swap for debias."""
    h = int(hashlib.md5(f"{pair_id}|{vp}".encode()).hexdigest(), 16)
    return (h % 2) == 1


# ---------- mechanical tiebreaker (LLM-free, defect-aware) ----------
def _feat(p):
    """Return (aspect=height/width, normalized 48x48 histogram, big_box_frac)."""
    from PIL import Image
    im = Image.open(p).convert("RGB")
    aspect = im.height / max(1, im.width)
    hist = im.resize((48, 48)).histogram()
    s = sum(hist) or 1
    hist = [x / s for x in hist]
    big_box = _big_uniform_box_frac(im)
    return aspect, hist, big_box


def _big_uniform_box_frac(im):
    """Approximate the largest fraction of the page occupied by a near-uniform
    (solid) horizontal band — a proxy for a LARGE broken/placeholder image box.
    Small footer glyphs / icons stay well below the 0.08 threshold used later."""
    try:
        from PIL import Image
        small = im.resize((32, 96))  # 32 wide, 96 tall rows
        px = small.load()
        w, h = small.size
        best = 0
        row = 0
        # count contiguous rows that are internally near-uniform AND extreme
        # (very dark or very light) — typical of black/white placeholder boxes.
        run = 0
        for y in range(h):
            vals = [px[x, y] for x in range(w)]
            mean = tuple(sum(v[c] for v in vals) / w for c in range(3))
            var = sum(sum((v[c] - mean[c]) ** 2 for c in range(3)) for v in vals) / w
            lum = sum(mean) / 3
            uniform = var < 120 and (lum < 32 or lum > 224)
            if uniform:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best / h
    except Exception:
        return 0.0


def _dist(fp, fc):
    """Distance of candidate feature fc from prototype fp.

    Height (vertical scale) is the primary signal; color histogram is a weak
    secondary tiebreaker. A LARGE near-uniform box (broken/placeholder image)
    that the prototype does not have adds a small penalty — but only when it is
    large (>8% of page), so tiny footer icons never contribute.
    """
    ap, hp, bp = fp
    ac, hc, bc = fc
    hgt = abs(math.log((ac + 1e-6) / (ap + 1e-6)))          # ~0..~1+
    hd = sum(abs(x - y) for x, y in zip(hp, hc))            # ~0..2
    # only penalize a large box the candidate has beyond what proto has
    box_excess = max(0.0, (bc - bp) - 0.08)
    return 2.0 * hgt + 0.4 * hd + 1.5 * box_excess


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
        cp = W / f'{e["id"]}_{vp}.jpg'
        if not cp.exists():
            make_strip(proto_path, left_path, right_path, w, cp)
        content = [{"type": "text", "text": PROMPT},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(cp)}}]
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
                # tie or all-abstain -> defect-aware mechanical last resort
                proto_paths = {vp: DS / e["task"] / "prototypes" / f"{vp}.jpg" for vp, _ in VIEWPORTS}
                a_paths = {vp: R / e["A"] / f"{vp}.png" for vp, _ in VIEWPORTS}
                b_paths = {vp: R / e["B"] / f"{vp}.png" for vp, _ in VIEWPORTS}
                winner = mech_verdict(proto_paths, a_paths, b_paths)
            rec = {"id": e["id"], "winner": winner}
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
