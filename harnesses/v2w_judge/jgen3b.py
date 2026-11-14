#!/usr/bin/env python3
"""JudgeHarness jgen3b — desktop dual-order debiasing (position-consistent = trustworthy) + tablet/mobile stacked responsive check + defect-aware mechanical fallback.

Task: given an unlabeled pair (task name + two candidate render directories
A/B), decide which candidate is visually closer to the prototype screenshots.
Writes verdicts.jsonl.

Evolution hard constraints:
1. Only the single model named by --model may be called;
2. No official score/answer files may be read at runtime;
3. Zero task-specific content in the code;
4. At most 3 LLM calls per pair judgement (here exactly 3).

Changes over jgen2 (targeting the small-gap bin; materially different from
jgen3's "prompt down-weighting + threshold" route):
- Observation: small-gap misjudgments are often "near ties" where the two
  candidates share nearly identical structure, and the decision gets pushed
  wrong by position bias + a single over-penalized decorative flaw.
  Per-viewport structural majority voting has almost no discriminative power
  on such pairs.
- Core change = a change in evidence aggregation, not prompt calibration:
  * Reallocate the 3-call budget: the desktop viewport (richest in
    information) gets dual-order (A|B and B|A) debiasing;
    both orders agree -> high-confidence strong vote (weight 2); orders
    conflict -> position-sensitive, this viewport abstains (mutual cancellation).
  * The 3rd call = a responsive check over tablet+mobile vertically stacked
    into a single image (weight 1), only actually deciding the outcome when
    desktop conflicts or one side is absent.
  * The defect-aware prompt & mechanical fallback carry over from jgen2
    (already validated; calibration untouched, avoiding the jgen3 regression).
"""
import argparse, base64, hashlib, io, json, math, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# desktop dual-order + (tablet, mobile) stacked check
DESKTOP = ("desktop", 520)
STACK_VPS = (("tablet", 340), ("mobile", 340))

PROMPT_SINGLE = """You will see one horizontal strip laid out as [ PROTOTYPE | X | Y ] for a single viewport.
The PROTOTYPE is the design target. X and Y are two candidate implementations.
Decide which candidate matches the PROTOTYPE better for THIS viewport.

Judge by, in rough priority order:
1. Overall layout, section structure and section ORDER (header, hero/content, sidebar, footer).
2. Presence and correct placement of the major blocks (nav bar, main text, images, footer columns).
3. Dominant colors / background / header & footer theme.

IMPORTANT — treat these as making a candidate a WORSE match, even if it otherwise looks cleaner:
- Broken images, black/placeholder image boxes, or missing images that the prototype has.
- Grossly OVERSIZED or MISPLACED images/icons (e.g. giant icons that blow up a section).
- A page that is far too TALL or too SHORT compared to the prototype (wrong vertical scale).
- Missing or truncated navigation / footer.

Do NOT reward a candidate merely for looking clean or having a pleasing color if it drops
sections, breaks images, or badly distorts the vertical proportions of the prototype.

Answer STRICT JSON only: {"winner": "X"} or {"winner": "Y"} or {"winner": "tie"}"""

PROMPT_STACK = """You will see ONE image containing TWO stacked horizontal strips:
- TOP strip  = TABLET viewport, laid out as [ PROTOTYPE | X | Y ].
- BOTTOM strip = MOBILE viewport, laid out as [ PROTOTYPE | X | Y ].
The PROTOTYPE is the design target. X and Y are two candidate implementations.
Decide which candidate, ACROSS BOTH viewports overall, matches the PROTOTYPE better
in responsive layout.

Judge by, in rough priority order:
1. Overall layout, section structure and section ORDER at each viewport.
2. Correct responsive reflow of the major blocks (nav, main content, images, footer).
3. Dominant colors / background / header & footer theme.

IMPORTANT — treat these as making a candidate a WORSE match, even if it otherwise looks cleaner:
- Broken images, black/placeholder image boxes, or missing images the prototype has.
- Grossly OVERSIZED or MISPLACED images/icons.
- A page far too TALL or too SHORT vs the prototype (wrong vertical scale).
- Missing or truncated navigation / footer.

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


def _strip_image(proto_p, left_p, right_p, panel_w, label):
    """Return a PIL strip image [PROTO|X|Y] with a viewport label band on top."""
    from PIL import Image, ImageDraw
    ims = [load_img(proto_p, panel_w), load_img(left_p, panel_w), load_img(right_p, panel_w)]
    band = 26
    H = max(im.height for im in ims) + band
    canvas = Image.new("RGB", (panel_w * 3 + 24, H), (24, 24, 24))
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 5), label, fill=(120, 220, 255))
    for i, (im, name) in enumerate(zip(ims, ("PROTOTYPE", "X", "Y"))):
        x = i * (panel_w + 12)
        dr.text((x + panel_w // 2 - 24, 5), name, fill=(255, 230, 90))
        canvas.paste(im, (x, band))
    return canvas


def make_stack(proto_map, left_map, right_map, out_p):
    """Vertically stack tablet strip over mobile strip into one image."""
    from PIL import Image
    parts = []
    for vp, w in STACK_VPS:
        parts.append(_strip_image(proto_map[vp], left_map[vp], right_map[vp], w,
                                  f"[{vp.upper()}]"))
    W = max(p.width for p in parts)
    gap = 8
    Htot = sum(p.height for p in parts) + gap * (len(parts) - 1)
    canvas = Image.new("RGB", (W, Htot), (10, 10, 10))
    y = 0
    for p in parts:
        canvas.paste(p, (0, y))
        y += p.height + gap
    canvas.save(out_p, "JPEG", quality=88)
    return out_p


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def _swap(pair_id, tag):
    """Deterministic (not PYTHONHASHSEED-dependent) left/right swap for debias."""
    h = int(hashlib.md5(f"{pair_id}|{tag}".encode()).hexdigest(), 16)
    return (h % 2) == 1


# ---------- mechanical tiebreaker (LLM-free, defect-aware) ----------
def _feat(p):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    aspect = im.height / max(1, im.width)
    hist = im.resize((48, 48)).histogram()
    s = sum(hist) or 1
    hist = [x / s for x in hist]
    return aspect, hist


def _dist(fp, fc):
    ap, hp = fp
    ac, hc = fc
    hgt = abs(math.log((ac + 1e-6) / (ap + 1e-6)))
    hd = sum(abs(x - y) for x, y in zip(hp, hc))
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


def _parse(raw):
    m = re.search(r'"winner"\s*:\s*"(X|Y|tie)"', raw or "")
    if not m:
        m = re.search(r'\b(X|Y|tie)\b', raw or "")
    if not m:
        return None
    tok = m.group(1)
    return None if tok == "tie" else tok


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

    def _call(strip_path, prompt):
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(strip_path)}}]
        r = client.chat.completions.create(model=args.model, temperature=0.0,
                                           messages=[{"role": "user", "content": content}])
        return _parse(r.choices[0].message.content or "")

    def ask_desktop(e, order_tag, x_is_a):
        """One desktop call. x_is_a: True -> X=A,Y=B ; False -> X=B,Y=A.
        Returns 'A'/'B'/None mapped to actual candidate."""
        vp, w = DESKTOP
        left_cand, right_cand = ("A", "B") if x_is_a else ("B", "A")
        left_path = R / e[left_cand] / f"{vp}.png"
        right_path = R / e[right_cand] / f"{vp}.png"
        proto_path = DS / e["task"] / "prototypes" / f"{vp}.jpg"
        cp = W / f'{e["id"]}_{vp}_{order_tag}.jpg'
        if not cp.exists():
            make_strip(proto_path, left_path, right_path, w, cp)
        tok = _call(cp, PROMPT_SINGLE)
        if tok is None:
            return None
        return left_cand if tok == "X" else right_cand

    def ask_stack(e):
        """One call over stacked tablet+mobile. Returns 'A'/'B'/None."""
        swap = _swap(e["id"], "stack")
        left_cand, right_cand = ("B", "A") if swap else ("A", "B")
        proto_map = {vp: DS / e["task"] / "prototypes" / f"{vp}.jpg" for vp, _ in STACK_VPS}
        left_map = {vp: R / e[left_cand] / f"{vp}.png" for vp, _ in STACK_VPS}
        right_map = {vp: R / e[right_cand] / f"{vp}.png" for vp, _ in STACK_VPS}
        cp = W / f'{e["id"]}_stack.jpg'
        if not cp.exists():
            make_stack(proto_map, left_map, right_map, cp)
        tok = _call(cp, PROMPT_STACK)
        if tok is None:
            return None
        return left_cand if tok == "X" else right_cand

    def run(e):
        try:
            # ---- 3 LLM calls: desktop x2 (order-debias) + stacked responsive x1 ----
            try:
                d1 = ask_desktop(e, "o1", True)    # X=A, Y=B
            except Exception:
                d1 = None
            try:
                d2 = ask_desktop(e, "o2", False)   # X=B, Y=A
            except Exception:
                d2 = None
            try:
                resp = ask_stack(e)
            except Exception:
                resp = None

            pts = {"A": 0.0, "B": 0.0}
            if d1 in ("A", "B") and d1 == d2:
                # position-consistent desktop verdict -> high confidence
                pts[d1] += 2.0
            else:
                # conflicting or single-sided desktop votes: each counts weakly,
                # a genuine order-conflict (d1 != d2) cancels out.
                for v in (d1, d2):
                    if v in ("A", "B"):
                        pts[v] += 1.0
            if resp in ("A", "B"):
                pts[resp] += 1.0

            if pts["A"] > pts["B"]:
                winner = "A"
            elif pts["B"] > pts["A"]:
                winner = "B"
            else:
                proto_paths = {vp: DS / e["task"] / "prototypes" / f"{vp}.jpg"
                               for vp in ("desktop", "tablet", "mobile")}
                a_paths = {vp: R / e["A"] / f"{vp}.png" for vp in ("desktop", "tablet", "mobile")}
                b_paths = {vp: R / e["B"] / f"{vp}.png" for vp in ("desktop", "tablet", "mobile")}
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
