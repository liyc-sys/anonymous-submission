#!/usr/bin/env python3
"""JudgeHarness night3 jgen — order-swap debias + mechanical tiebreak.

Materially different from per-viewport-majority: this genome keeps whole-page
three-viewport single-call judging, but debiases via the two display orders
A-B / B-A, and uses a deterministic color-histogram + aspect-ratio
mechanical signal as a "tie-breaking-only" adjudicator (clamped to ±0.9,
never overriding a unanimous vote).

Hard constraints: only call the model named by --model; read no official
answers; zero task-specific; <=3 LLM calls/pair;
single file; keep the CLI and verdicts.jsonl format; idempotently skip already-judged ids.
"""
import argparse, base64, json, math, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = """You will see, for each of three viewports (desktop / tablet / mobile), one strip image laid out as [ PROTOTYPE | CANDIDATE A | CANDIDATE B ].
The PROTOTYPE is the design target. Decide which candidate matches the prototype better overall (layout, sections, images, text, colors).
Answer STRICT JSON only: {"winner": "A"} or {"winner": "B"} or {"winner": "tie"}"""

VPS = (("desktop", 330, 1.0), ("tablet", 300, 0.6), ("mobile", 240, 0.5))

# ---------- image / strip helpers ----------
def load_img(p, width):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    im = im.resize((width, max(1, round(im.height * width / im.width))))
    if im.height > 1800:
        im = im.crop((0, 0, im.width, 1800))
    return im

def strip3(proto_p, first_p, second_p, panel_w, out_p):
    """first/second are the DISPLAYED candidates (order may be swapped)."""
    from PIL import Image, ImageDraw
    ims = [load_img(proto_p, panel_w), load_img(first_p, panel_w), load_img(second_p, panel_w)]
    H = max(im.height for im in ims) + 26
    canvas = Image.new("RGB", (panel_w * 3 + 24, H), (24, 24, 24))
    dr = ImageDraw.Draw(canvas)
    for i, (im, name) in enumerate(zip(ims, ("PROTOTYPE", "A", "B"))):
        x = i * (panel_w + 12)
        dr.text((x + panel_w // 2 - 24, 5), name, fill=(255, 230, 90))
        canvas.paste(im, (x, 26))
    canvas.save(out_p, "JPEG", quality=68)
    return out_p

def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()

# ---------- deterministic mechanical signal ----------
def _hist(p, bins=4):
    from PIL import Image
    small = Image.open(p).convert("RGB").resize((64, 64))
    h = [0] * (bins ** 3)
    step = 256 // bins
    for r, g, bl in small.getdata():
        h[min(r // step, bins-1) * bins * bins + min(g // step, bins-1) * bins + min(bl // step, bins-1)] += 1
    n = sum(h) or 1
    return [c / n for c in h]

def _aspect(p):
    from PIL import Image
    w, ht = Image.open(p).size
    return ht / max(1, w)

def _l1(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))

def mech_signal(proto, a, b):
    """Positive => A closer to prototype, negative => B. Clamped to [-0.9,0.9]."""
    total = 0.0
    for vp, _w, wgt in VPS:
        try:
            pp = proto / f"{vp}.jpg"
            ap = a / f"{vp}.png"
            bp = b / f"{vp}.png"
            hp, ha, hb = _hist(pp), _hist(ap), _hist(bp)
            dca = _l1(hp, ha); dcb = _l1(hp, hb)           # color distance (0..2)
            arp = _aspect(pp); ara = _aspect(ap); arb = _aspect(bp)
            dha = abs(math.log((ara + 1e-6) / (arp + 1e-6)))  # shape/height distance
            dhb = abs(math.log((arb + 1e-6) / (arp + 1e-6)))
            da = 1.0 * dca + 0.7 * dha
            db = 1.0 * dcb + 0.7 * dhb
            total += wgt * (db - da)                        # >0 favors A
        except Exception:
            continue
    return max(-0.9, min(0.9, total * 1.5))

# ---------- LLM ----------
def ask(client, model, imgs):
    content = [{"type": "text", "text": PROMPT}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(p)}} for p in imgs]
    for _ in range(2):
        r = client.chat.completions.create(model=model, temperature=0.0,
                                           messages=[{"role": "user", "content": content}])
        raw = r.choices[0].message.content or ""
        m = re.search(r'"winner"\s*:\s*"(A|B|tie)"', raw)
        if m:
            return m.group(1)
    return "tie"

def _vote(w):
    return {"A": 1.0, "B": -1.0, "tie": 0.0}[w]

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

    def run(e):
        try:
            proto = Path(args.datasets) / e["task"] / "prototypes"
            aA, aB = R / e["A"], R / e["B"]
            # build both-order strips (cache on disk)
            imgs_ab, imgs_ba = [], []
            for vp, w, _wgt in VPS:
                p_ab = W / f'{e["id"]}_{vp}_ab.jpg'
                p_ba = W / f'{e["id"]}_{vp}_ba.jpg'
                if not p_ab.exists():
                    strip3(proto / f"{vp}.jpg", aA / f"{vp}.png", aB / f"{vp}.png", w, p_ab)
                if not p_ba.exists():
                    strip3(proto / f"{vp}.jpg", aB / f"{vp}.png", aA / f"{vp}.png", w, p_ba)
                imgs_ab.append(p_ab); imgs_ba.append(p_ba)

            # mechanical signal (no LLM cost); positive => A
            mech = mech_signal(proto, aA, aB)

            # two LLM calls, swapped display order
            w_ab = ask(client, args.model, imgs_ab)                     # displayed A == actual A
            w_ba_disp = ask(client, args.model, imgs_ba)               # displayed A == actual B
            w_ba = {"A": "B", "B": "A", "tie": "tie"}[w_ba_disp]       # remap to actual

            # reconcile: unanimous LLM (=±2) can't be flipped by mech(±0.9)
            score = _vote(w_ab) + _vote(w_ba) + mech
            if score > 1e-9:
                winner = "A"
            elif score < -1e-9:
                winner = "B"
            else:
                # dead tie: prefer forward LLM else mech sign
                winner = w_ab if w_ab != "tie" else ("A" if mech >= 0 else "B")

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
