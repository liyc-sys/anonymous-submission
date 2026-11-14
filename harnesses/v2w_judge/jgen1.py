#!/usr/bin/env python3
"""JudgeHarness jgen1 — per-viewport high-resolution independent voting + position debiasing + mechanical tiebreak.

Task: given an unlabeled pair (task name + two candidate render directories
A/B), decide which candidate is visually closer to the prototype screenshots.
Writes verdicts.jsonl.

Evolution hard constraints:
1. Only the single model named by --model may be called;
2. No official score/answer files may be read at runtime;
3. Zero task-specific content in the code;
4. At most 3 LLM calls per pair judgement (here exactly 3: one per viewport).

Approach:
- The old version crammed the 3-viewport triptych strips into a single call, each
  panel only 330px, losing all detail and failing easily.
- The new version makes one call per viewport with wider panels (480/380/300),
  preserving detail;
- Three-viewport majority voting suppresses single-viewport noise;
- Deterministic left/right swap per (id, viewport) removes position bias;
- Only when the vote is tied or all abstain do the mechanical features (histogram
  + aspect-ratio distance) adjudicate; never as a hard override.
"""
import argparse, base64, hashlib, io, json, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIEWPORTS = (("desktop", 480), ("tablet", 380), ("mobile", 300))

PROMPT = """You will see one horizontal strip laid out as [ PROTOTYPE | X | Y ] for a single viewport.
The PROTOTYPE is the design target. X and Y are two candidate implementations.
Decide which candidate matches the PROTOTYPE better for THIS viewport, considering overall layout,
section structure & order, presence/placement of images, text blocks, backgrounds and colors.
Answer STRICT JSON only: {"winner": "X"} or {"winner": "Y"} or {"winner": "tie"}"""


def load_img(p, width):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    im = im.resize((width, max(1, round(im.height * width / im.width))))
    if im.height > 2000:
        im = im.crop((0, 0, im.width, 2000))
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
    canvas.save(out_p, "JPEG", quality=82)
    return out_p


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def _swap(pair_id, vp):
    """Deterministic (not PYTHONHASHSEED-dependent) left/right swap for debias."""
    h = int(hashlib.md5(f"{pair_id}|{vp}".encode()).hexdigest(), 16)
    return (h % 2) == 1


# ---------- mechanical tiebreaker (LLM-free) ----------
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
    hd = sum(abs(x - y) for x, y in zip(hp, hc))          # ~0..6
    ad = abs(ap - ac) / max(ap, ac, 1e-6)                 # relative aspect gap
    return hd + 1.5 * ad


def mech_verdict(proto_paths, a_paths, b_paths):
    try:
        da = db = 0.0
        for vp in proto_paths:
            fp = _feat(proto_paths[vp])
            da += _dist(fp, _feat(a_paths[vp]))
            db += _dist(fp, _feat(b_paths[vp]))
        if abs(da - db) < 0.02 * max(da, db, 1e-6):
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
        # X panel = left, Y panel = right
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
        # X->left_cand, Y->right_cand
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
                # tie or all-abstain -> mechanical last resort
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
