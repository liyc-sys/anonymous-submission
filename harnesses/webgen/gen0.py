#!/usr/bin/env python3
"""MiniWebGen gen0 -- WebGen-Bench minimal webpage generation scaffold (this file IS the evolution genome).

Task: given a natural-language website requirement (actor_input.json, containing only instruction/Category/application_type),
generate a purely static website in the working directory (index.html as the entry), with working interactions and an appearance matching the requirement.

Hard constraints (must never be violated during evolution):
1. Only the one model specified by --model may be called; calling any other model is forbidden;
2. At runtime, never read any evaluation artifacts (ui_instruct / result.json / screenshots);
3. The code must not contain the name or page content of any specific task (it must be generic across all tasks);
4. Total LLM calls per task <= 20 (--max-steps default <= 18);
5. The output must be a purely static site: reference no external URLs (CDN/fonts/images), all interaction via inline or local JS.
"""
import argparse
import json
import re
import time
from pathlib import Path

import httpx

SYSTEM = """You are a web developer. Build a COMPLETE static website in the working directory that fulfils the user's requirement.

Rules:
- Entry point must be index.html. Use plain HTML/CSS/JavaScript only (no build step, no npm).
- The site must work OFFLINE: never reference any external URL (no CDN scripts, no web fonts, no remote images). Draw icons/graphics with CSS or inline SVG.
- Make the requested features actually WORK with JavaScript and plausible built-in mock data (e.g. search really filters, forms validate and show results, buttons respond).
- Follow explicit style requirements in the instruction (colors, layout) exactly.

You have these tools. Use one or more per reply:
<write path="relative/path">full file content</write>
<read path="relative/path"/>
<ls/>
<done/>

Write complete files (no diffs, no placeholders like TODO). When the site is complete and functional, output <done/>."""

RE_WRITE = re.compile(r'<write path="([^"]+)">(.*?)</write>', re.S)
RE_READ = re.compile(r'<read path="([^"]+)"\s*/>')
RE_LS = re.compile(r"<ls\s*/>")
RE_DONE = re.compile(r"<done\s*/>")


def call_llm(base, key, model, messages, temperature):
    body = {"model": model, "messages": messages, "stream": True, "temperature": temperature}
    parts = []
    with httpx.stream("POST", f"{base}/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json=body, timeout=600, trust_env=False) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)["choices"][0]["delta"]
            except Exception:
                continue
            parts.append(d.get("content") or "")
    return "".join(parts)


def safe(workdir, rel):
    p = (workdir / rel).resolve()
    if not str(p).startswith(str(workdir.resolve())):
        raise ValueError("path escapes workdir")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-json", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model", default="Qwen3.5-9B")
    ap.add_argument("--base-url", default="https://api.example.com/v1")
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    t0 = time.time()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    key = Path(args.api_key_file).read_text().strip()
    task = json.loads(Path(args.task_json).read_text())

    user0 = (f"Requirement:\n{task['instruction']}\n\n"
             f"Application type: {task.get('application_type', 'website')}\n\n"
             "Build the full site now. Start by writing index.html.")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user0}]

    calls, parse_misses, llm_errors, done = 0, 0, 0, False
    for step in range(args.max_steps):
        try:
            out = call_llm(args.base_url, key, args.model, messages, args.temperature)
            calls += 1
            if not out.strip():
                raise RuntimeError("empty response")
        except Exception:
            llm_errors += 1
            time.sleep(5)
            try:
                out = call_llm(args.base_url, key, args.model, messages, args.temperature)
                calls += 1
                if not out.strip():
                    raise RuntimeError("empty response")
            except Exception:
                break
        messages.append({"role": "assistant", "content": out})
        results, acted = [], False
        for m in RE_WRITE.finditer(out):
            acted = True
            try:
                p = safe(workdir, m.group(1))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(m.group(2), encoding="utf-8")
                results.append(f"[wrote {m.group(1)}: {len(m.group(2))} chars]")
            except Exception as e:
                results.append(f"[write {m.group(1)} failed: {e}]")
        for m in RE_READ.finditer(out):
            acted = True
            try:
                content = safe(workdir, m.group(1)).read_text(encoding="utf-8")[:4000]
                results.append(f"[{m.group(1)}]:\n{content}")
            except Exception as e:
                results.append(f"[read {m.group(1)} failed: {e}]")
        if RE_LS.search(out):
            acted = True
            files = [str(p.relative_to(workdir)) for p in workdir.rglob("*") if p.is_file()]
            results.append("[files]: " + ", ".join(sorted(files)[:50]))
        if RE_DONE.search(out):
            done = True
            break
        if not acted:
            parse_misses += 1
            results.append("[no tool call detected. Use <write path=...>...</write> or <done/>]")
        messages.append({"role": "user", "content": "\n".join(results)[:6000]})

    log_dir = workdir / ".agent"
    log_dir.mkdir(exist_ok=True)
    log_dir.joinpath("log.json").write_text(json.dumps({
        "calls": calls, "done": done, "seconds": round(time.time() - t0, 1),
        "index_exists": (workdir / "index.html").exists(),
        "parse_misses": parse_misses, "llm_errors": llm_errors,
    }, indent=1))


if __name__ == "__main__":
    main()
