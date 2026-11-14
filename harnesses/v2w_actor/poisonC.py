#!/usr/bin/env python3
"""PoisonC -- poisoning ablation C: context bloat (simulating OpenHands-style verbose history).
After every file write the full content is echoed back into the conversation, file reads are not truncated, every turn attaches the full directory listing, and feedback is not clipped.
Everything except the echo/truncation policy is byte-identical to gen0."""
import argparse, base64, io, json, re, time
from pathlib import Path

SYSTEM = """You are a web developer agent. Build a static webpage in the working directory that visually matches the three prototype screenshots (desktop 1920px / tablet 1024px / mobile 375px).

Rules:
- This environment is OFFLINE: external URLs never load. Use only files under resources/ (already present in your working directory) or inline SVG/CSS.
- The entry file must be index.html.
- Respond with one or more actions per turn, using EXACTLY this syntax:

<write path="index.html">
...complete file content...
</write>

<read path="style.css"/>

<ls/>

<done/>

- <write> replaces the whole file: always output complete file contents, never fragments or diffs.
- Make the page responsive so it matches all three screenshots at their widths.
- When you are satisfied the page is complete, output <done/>."""

def img_part(p, width, max_h):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    im = im.resize((width, max(1, round(im.height * width / im.width))))
    if im.height > max_h:
        im = im.crop((0, 0, im.width, max_h))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}

def list_dir(root, cap=200):
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(s.startswith(".") for s in p.parts):
            out.append(f"{p.relative_to(root)} ({p.stat().st_size}B)")
        if len(out) >= cap:
            out.append("...")
            break
    return "\n".join(out) or "(empty)"

def safe(workdir, rel):
    p = (workdir / rel).resolve()
    if not str(p).startswith(str(workdir.resolve())):
        raise ValueError("path escapes workdir")
    return p

RE_WRITE = re.compile(r'<write\s+path="([^"]+)"\s*>\n?(.*?)</write>', re.S)
RE_READ = re.compile(r'<read\s+path="([^"]+)"\s*/>')
RE_LS = re.compile(r"<ls\s*/>")
RE_DONE = re.compile(r"<done\s*/>")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model", default="Qwen3.5-9B")
    ap.add_argument("--base-url", default="http://127.0.0.1:8399/v1")
    ap.add_argument("--api-key-file", default="./.api_key")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()
    task_dir, workdir = Path(args.task_dir), Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log = {"calls": 0, "steps": [], "t0": time.time(), "poison": "C_context_bloat"}
    logf = workdir / ".agent" / "log.json"
    logf.parent.mkdir(exist_ok=True)

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=Path(args.api_key_file).read_text().strip(), timeout=300)

    def llm(messages):
        log["calls"] += 1
        r = client.chat.completions.create(model=args.model, messages=messages, temperature=args.temperature)
        return r.choices[0].message.content or ""

    proto = task_dir / "prototypes"
    first = [{"type": "text", "text":
              "Build the webpage matching these three prototype screenshots "
              "(order: desktop, tablet, mobile; tall pages are cropped - infer the rest).\n\n"
              f"Files available under resources/ in your working directory:\n{list_dir(workdir / 'resources')}\n\n"
              "Start by writing index.html (you may also write style.css / script.js)."}]
    for vp, w in (("desktop", 480), ("tablet", 440), ("mobile", 360)):
        p = proto / f"{vp}.jpg"
        if p.exists():
            first.append(img_part(p, w, 2600))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first}]

    done = False
    for step in range(args.max_steps):
        try:
            out = llm(messages)
        except Exception as e:
            log["steps"].append({"step": step, "error": f"llm: {type(e).__name__}: {e}"[:300]})
            time.sleep(5)
            continue
        messages.append({"role": "assistant", "content": out})
        results, acted = [], False
        for m in RE_WRITE.finditer(out):
            acted = True
            try:
                p = safe(workdir, m.group(1))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(m.group(2))
                results.append(f"[write ok: {m.group(1)} ({len(m.group(2))} chars)]\n"
                               f"=== CURRENT CONTENT OF {m.group(1)} (echoed for your reference) ===\n"
                               f"{m.group(2)[:60000]}\n=== END OF {m.group(1)} ===")
            except Exception as e:
                results.append(f"[write FAILED {m.group(1)}: {e}]")
        for m in RE_READ.finditer(out):
            acted = True
            try:
                content = safe(workdir, m.group(1)).read_text()[:60000]
                results.append(f"[read {m.group(1)}]:\n{content}")
            except Exception as e:
                results.append(f"[read FAILED {m.group(1)}: {e}]")
        if RE_LS.search(out):
            acted = True
            results.append("[ls]:\n" + list_dir(workdir))
        log["steps"].append({"step": step, "acted": acted, "n_out": len(out)})
        if RE_DONE.search(out):
            done = True
            break
        if not acted:
            results.append('No valid action found. Use exactly <write path="...">content</write>, '
                           '<read path="..."/>, <ls/> or <done/>.')
        # Context bloat: attach the full workspace listing + resource listing every turn (simulating framework-style redundant observation)
        results.append("=== WORKSPACE STATE (auto-attached every turn) ===\n" + list_dir(workdir))
        results.append("=== RESOURCES AVAILABLE (auto-attached every turn) ===\n" + list_dir(workdir / "resources"))
        messages.append({"role": "user", "content": "\n".join(results)[:120000]})

    if not (workdir / "index.html").exists():
        try:
            out = llm(messages + [{"role": "user", "content":
                     'index.html is still missing. Output the COMPLETE index.html now inside '
                     '<write path="index.html">...</write>.'}])
            m = RE_WRITE.search(out)
            if m:
                safe(workdir, "index.html").write_text(m.group(2))
        except Exception as e:
            log["final_error"] = str(e)[:300]
    log["done"] = done
    log["seconds"] = round(time.time() - log["t0"], 1)
    log["index_exists"] = (workdir / "index.html").exists()
    logf.write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(json.dumps({k: log[k] for k in ("calls", "done", "seconds", "index_exists")}))

if __name__ == "__main__":
    main()
