#!/usr/bin/env python3
"""PoisonB -- poisoning ablation B: replace gen0's loose XML tool protocol with a strict JSON single-action protocol
(simulating OpenHands-style function calling: only one action per turn, arguments must be valid JSON, large file content must be JSON-escaped).
Everything except the protocol (SYSTEM description + parser) is byte-identical to gen0."""
import argparse, base64, io, json, re, time
from pathlib import Path

SYSTEM = """You are a web developer agent. Build a static webpage in the working directory that visually matches the three prototype screenshots (desktop 1920px / tablet 1024px / mobile 375px).

Rules:
- This environment is OFFLINE: external URLs never load. Use only files under resources/ (already present in your working directory) or inline SVG/CSS.
- The entry file must be index.html.
- Respond with EXACTLY ONE action per turn, as a single valid JSON object inside a ```json code fence. The JSON must parse; all string values (including file content) must be properly JSON-escaped. No other action formats are accepted.

Available actions:

```json
{"action": "write_file", "arguments": {"path": "index.html", "content": "<!DOCTYPE html>..."}}
```

```json
{"action": "read_file", "arguments": {"path": "style.css"}}
```

```json
{"action": "list_files", "arguments": {}}
```

```json
{"action": "finish", "arguments": {}}
```

- write_file replaces the whole file: always provide complete file contents, never fragments or diffs.
- ONE action per turn. Additional JSON objects in the same reply are ignored.
- Make the page responsive so it matches all three screenshots at their widths.
- When you are satisfied the page is complete, emit the finish action."""

def img_part(p, width, max_h):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    im = im.resize((width, max(1, round(im.height * width / im.width))))
    if im.height > max_h:
        im = im.crop((0, 0, im.width, max_h))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}

def list_dir(root, cap=40):
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

RE_FENCE = re.compile(r"```json\s*\n(.*?)```", re.S)

def parse_action(out):
    """Strict parsing: first look for a ```json fence, then fall back to a bare {...}; a json.loads failure is a failure. Only the first one is taken."""
    m = RE_FENCE.search(out)
    blob = m.group(1) if m else None
    if blob is None:
        b0, b1 = out.find("{"), out.rfind("}")
        if b0 >= 0 and b1 > b0:
            blob = out[b0:b1 + 1]
    if blob is None:
        raise ValueError("no JSON object found")
    obj = json.loads(blob)
    if not isinstance(obj, dict) or "action" not in obj:
        raise ValueError("JSON must be an object with an 'action' field")
    return obj.get("action"), obj.get("arguments") or {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model", default="Qwen3.5-9B")
    ap.add_argument("--base-url", default="http://127.0.0.1:8399/v1")
    ap.add_argument("--api-key-file", default="./.api_key")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()
    task_dir, workdir = Path(args.task_dir), Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log = {"calls": 0, "steps": [], "t0": time.time(), "poison": "B_strict_json_protocol"}
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
        try:
            action, arguments = parse_action(out)
            acted = True
            if action == "write_file":
                p = safe(workdir, arguments["path"])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(arguments.get("content", ""))
                results.append(f'[write ok: {arguments["path"]} ({len(arguments.get("content", ""))} chars)]')
            elif action == "read_file":
                content = safe(workdir, arguments["path"]).read_text()[:4000]
                results.append(f'[read {arguments["path"]}]:\n{content}')
            elif action == "list_files":
                results.append("[ls]:\n" + list_dir(workdir))
            elif action == "finish":
                done = True
            else:
                acted = False
                results.append(f"[error: unknown action {action!r}]")
        except Exception as e:
            results.append(f"[error: invalid action JSON: {str(e)[:120]}. "
                           "Reply with exactly ONE valid JSON action in a ```json fence.]")
        log["steps"].append({"step": step, "acted": acted, "n_out": len(out)})
        if done:
            break
        messages.append({"role": "user", "content": "\n".join(results)[:6000]})

    if not (workdir / "index.html").exists():
        try:
            out = llm(messages + [{"role": "user", "content":
                     'index.html is still missing. Emit ONE write_file action with the COMPLETE index.html now.'}])
            try:
                action, arguments = parse_action(out)
                if action == "write_file":
                    safe(workdir, arguments["path"]).write_text(arguments.get("content", ""))
            except Exception:
                pass
        except Exception as e:
            log["final_error"] = str(e)[:300]
    log["done"] = done
    log["seconds"] = round(time.time() - log["t0"], 1)
    log["index_exists"] = (workdir / "index.html").exists()
    log["parse_misses"] = sum(1 for s in log["steps"] if not s.get("acted") and not s.get("error"))
    logf.write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(json.dumps({k: log[k] for k in ("calls", "done", "seconds", "index_exists", "parse_misses")}))

if __name__ == "__main__":
    main()
