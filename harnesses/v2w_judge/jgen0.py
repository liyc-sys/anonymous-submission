#!/usr/bin/env python3
"""JudgeHarness gen0 — minimal judging scaffold (the evolution genome on the judge side).

Task: given an exam (unlabeled pairs: task name + render directories of two
candidate pages A/B), decide which candidate is visually closer to the task's
prototype screenshot. Writes verdicts.jsonl.

Evolution hard constraints:
1. Only the single model named by --model may be called;
2. No official score/answer files may be read at runtime (the answer book is invisible to this program);
3. Zero task-specific content in the code;
4. At most 3 LLM calls per pair judgement.
"""
import argparse, base64, io, json, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = """You will see, for each of three viewports (desktop / tablet / mobile), one strip image laid out as [ PROTOTYPE | CANDIDATE A | CANDIDATE B ].
The PROTOTYPE is the design target. Decide which candidate matches the prototype better overall (layout, sections, images, text, colors).
Answer STRICT JSON only: {"winner": "A"} or {"winner": "B"} or {"winner": "tie"}"""

def load_img(p, width):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    im = im.resize((width, max(1, round(im.height * width / im.width))))
    if im.height > 1800:
        im = im.crop((0, 0, im.width, 1800))
    return im

def strip3(proto_p, a_p, b_p, panel_w, out_p):
    from PIL import Image, ImageDraw
    ims = [load_img(proto_p, panel_w), load_img(a_p, panel_w), load_img(b_p, panel_w)]
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
            imgs = []
            for vp, w in (("desktop", 330), ("tablet", 300), ("mobile", 240)):
                cp = W / f'{e["id"]}_{vp}.jpg'
                if not cp.exists():
                    strip3(Path(args.datasets) / e["task"] / "prototypes" / f"{vp}.jpg",
                           R / e["A"] / f"{vp}.png", R / e["B"] / f"{vp}.png", w, cp)
                imgs.append(cp)
            content = [{"type": "text", "text": PROMPT}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(p)}} for p in imgs]
            winner, raw = "tie", ""
            for attempt in range(2):
                r = client.chat.completions.create(model=args.model, temperature=0.0,
                                                   messages=[{"role": "user", "content": content}])
                raw = r.choices[0].message.content or ""
                m = re.search(r'"winner"\s*:\s*"(A|B|tie)"', raw)
                if m:
                    winner = m.group(1)
                    break
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
