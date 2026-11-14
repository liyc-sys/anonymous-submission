#!/usr/bin/env python3
"""MiniWebGen gen-wire -- WebGen-Bench static webpage generation scaffold (evolution genome).

This generation (relative to the gen0 champion) changes only one dimension: **interaction-wiring integrity**.
The generic evidenced failure mode is secondary views being unreachable, script exceptions blanking the page, and forms with no visible feedback after submission.
Countermeasures (a different dimension from the "final feature self-check list" candidate):
  1. The system prompt now pushes **multi-page + real <a href> native navigation + one try/catch per widget**, so one script crash cannot blank the whole page;
  2. After generation, run a **deterministic static wiring checkup**: scan every html file's internal links and anchors; dangling links / missing anchors -> reject <done/>,
     feeding the concrete gaps back to the model to fix (including an "unwired form" warning); bounded to 3 rounds, never burning through the step budget.
The tool protocol is untouched, not one word of it (lesson from gen1a: rewording breaks the XML-parsing fit).

Hard constraints unchanged: only call --model; read no evaluation artifacts; generic, no task keywords; <=20 calls; purely static, external links forbidden.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import httpx

SYSTEM = """You are a web developer. Build a COMPLETE, FUNCTIONAL static website in the working directory that fulfils the user's requirement.

Architecture (this directly affects whether features are reachable):
- Prefer a MULTI-PAGE site: put each major feature/section on its OWN .html file (index.html plus one file per feature). A very small site may stay single-page.
- Put an IDENTICAL navigation bar with real <a href="..."> links to EVERY page at the top of EVERY page, so every feature is reachable by clicking a link.
- Navigation MUST use real hyperlinks: href to another .html file, or href="#id" pointing at a section that actually exists on that page. Do NOT rely on JavaScript alone to swap views — if a script throws, the page must still show its content.
- Wrap each independent widget's JavaScript in its OWN try/catch so one broken feature can never blank out the whole page. Real content must exist in the HTML before any JS runs.

Functionality:
- Make requested features actually WORK with JS and plausible built-in mock data: search really filters and renders results, filters/buttons update visible content, detail views show real fields.
- Every <form> must call event.preventDefault() and then render a visible success/result message into an element on the page (never a bare reload, never a silent no-op).

Rules:
- Entry point must be index.html. Use plain HTML/CSS/JavaScript only (no build step, no npm).
- The site must work OFFLINE: never reference any external URL (no CDN scripts, no web fonts, no remote images). Draw icons/graphics with CSS or inline SVG.
- Follow explicit style requirements exactly, translating requested color names into valid CSS values.

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

RE_HREF = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.I)
RE_ID = re.compile(r'\bid\s*=\s*["\']([^"\']+)["\']', re.I)
RE_NAME = re.compile(r'\bname\s*=\s*["\']([^"\']+)["\']', re.I)

MAX_FIX_ROUNDS = 3


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


def analyze_site(workdir):
    """Static interaction-wiring audit. Returns dict of concrete, mechanically-found gaps.
    No requirement text is consulted -> fully generic."""
    html_files = {}
    js_blob = []
    for p in workdir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(workdir).as_posix()
        if rel.startswith(".agent/"):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if rel.lower().endswith((".html", ".htm")):
            html_files[rel] = txt
            js_blob.append(txt)
        elif rel.lower().endswith(".js"):
            js_blob.append(txt)
    global_js = "\n".join(js_blob).lower().replace('"', "'")
    global_has_submit = ("preventdefault" in global_js or
                         "addeventlistener('submit'" in global_js or
                         "onsubmit" in global_js)

    ids = {}
    for rel, c in html_files.items():
        s = set(RE_ID.findall(c)) | set(RE_NAME.findall(c))
        ids[rel] = s

    missing_pages, missing_anchors, unwired_forms = set(), set(), []
    for rel, c in html_files.items():
        cdir = os.path.dirname(rel)
        for href in RE_HREF.findall(c):
            h = href.strip()
            if not h:
                continue
            low = h.lower()
            if low.startswith("#"):
                frag = h[1:]
                if frag and frag not in ids.get(rel, set()):
                    missing_anchors.add(f"{rel}#{frag}")
                continue
            if low.startswith(("http://", "https://", "//", "mailto:",
                               "tel:", "javascript:", "data:")):
                continue
            path, frag = h, None
            if "#" in h:
                path, frag = h.split("#", 1)
            path = path.split("?")[0]
            if path == "":
                tgt = rel
            else:
                base = os.path.basename(path)
                # skip asset links (.css/.js/.png/...) - only page navigation matters
                if "." in base and not path.lower().endswith((".html", ".htm")):
                    continue
                if path.endswith("/"):
                    path = path + "index.html"
                tgt = os.path.normpath(os.path.join(cdir, path)).replace("\\", "/")
                if tgt not in html_files:
                    missing_pages.add(tgt)
                    continue
            if frag and frag not in ids.get(tgt, set()):
                missing_anchors.add(f"{tgt}#{frag}")
        if re.search(r"<form", c, re.I) and not global_has_submit:
            unwired_forms.append(rel)

    return {
        "pages": sorted(html_files.keys()),
        "missing_pages": sorted(missing_pages),
        "missing_anchors": sorted(missing_anchors),
        "unwired_forms": sorted(set(unwired_forms)),
        "index_exists": "index.html" in html_files,
    }


def build_fix_message(rep):
    lines = ["Not done yet - a static navigation/wiring check found broken interactions. "
             "Fix ALL of these, then output <done/> again:"]
    if not rep["index_exists"]:
        lines.append("- index.html is missing. Create it as the entry page with the shared nav bar.")
    if rep["missing_pages"]:
        lines.append("- These pages are LINKED but do not exist. Create each one as a real, "
                     "functional page with the same shared nav bar and real content: "
                     + ", ".join(rep["missing_pages"]))
    if rep["missing_anchors"]:
        lines.append("- These links point to sections/ids that DO NOT EXIST (dead nav -> blank view). "
                     "Either add an element with that id, or fix the href: "
                     + ", ".join(rep["missing_anchors"]))
    if rep["unwired_forms"]:
        lines.append("- These pages contain a <form> with no submit handler anywhere. Add an onsubmit "
                     "handler that calls event.preventDefault() and shows a visible result message: "
                     + ", ".join(rep["unwired_forms"]))
    return "\n".join(lines)


def has_blocking_gap(rep):
    return (not rep["index_exists"] or rep["missing_pages"]
            or rep["missing_anchors"] or rep["unwired_forms"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-json", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model", default="Qwen3.5-9B")
    ap.add_argument("--base-url", default="https://api.example.com/v1")
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    t0 = time.time()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    key = Path(args.api_key_file).read_text().strip()
    task = json.loads(Path(args.task_json).read_text())

    user0 = (f"Requirement:\n{task['instruction']}\n\n"
             f"Application type: {task.get('application_type', 'website')}\n\n"
             "Build the full site now. Start by writing index.html with the shared navigation bar.")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user0}]

    calls = parse_misses = llm_errors = link_fix_rounds = 0
    done = False
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
            acted = True
            rep = analyze_site(workdir)
            if has_blocking_gap(rep) and link_fix_rounds < MAX_FIX_ROUNDS and step < args.max_steps - 1:
                link_fix_rounds += 1
                fb = build_fix_message(rep)
                if results:
                    fb = "\n".join(results)[:2000] + "\n\n" + fb
                messages.append({"role": "user", "content": fb[:6000]})
                continue
            done = True
            break

        if not acted:
            parse_misses += 1
            results.append("[no tool call detected. Use <write path=...>...</write> or <done/>]")
        messages.append({"role": "user", "content": "\n".join(results)[:6000]})

    final = analyze_site(workdir)
    missing_final = (len(final["missing_pages"]) + len(final["missing_anchors"])
                     + (0 if final["index_exists"] else 1))

    log_dir = workdir / ".agent"
    log_dir.mkdir(exist_ok=True)
    log_dir.joinpath("log.json").write_text(json.dumps({
        "calls": calls, "done": done, "seconds": round(time.time() - t0, 1),
        "index_exists": (workdir / "index.html").exists(),
        "parse_misses": parse_misses, "llm_errors": llm_errors,
        "link_checks_ran": True,
        "pages_generated": len(final["pages"]),
        "missing_links_final": missing_final,
        "unwired_forms_final": len(final["unwired_forms"]),
        "link_fix_rounds": link_fix_rounds,
    }, indent=1))


if __name__ == "__main__":
    main()
