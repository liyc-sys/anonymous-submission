#!/usr/bin/env python3
"""MiniWebGen gen-wire+depth+persist+navland+indexfloor -- WebGen-Bench static webpage generation scaffold (evolution genome).

This generation advances a single dimension relative to the champion (multi-page native links + deterministic wiring audit
+ persistence read-back + nav landing heading verification): **a self-sufficient index.html as the "site never materialized"
floor (single self-contained index as site-missing floor)**.
The generic evidenced failure mode: the most catastrophic and previously unhandled class is "the whole site never materialized" —
during generation index.html simply never lands on disk, or only an empty shell lands that waits for later files to fill it,
so when the UI agent opens the browser the "site is missing", the entire case scores all NONE, and any navigation/heading
polish is moot. Two previously lost candidates (route-list self-repair / multi-page emission integrity guard) both
"repair fragile multi-page emission after the fact"; the materially different mechanism here changes the generation "target":
front-load the model to always emit a self-sufficient index.html first, writing each task-named block directly into it as an
in-page anchor block, with extra .html files as optional enhancement only; plus a deterministic post-hoc check: if index.html
is missing / below the byte floor / below the visible-text floor, reject <done/> and demand it be completed.
This strikes the disk-materialization root cause rather than route wiring, with low regression: a non-empty index is strictly
better than nothing on NONE cases, while already-passing multi-page sites (index already substantial) are entirely unaffected.
The tool protocol is untouched, not one word. Hard constraints unchanged: only call --model; read no evaluation artifacts; generic, no keywords; <=20 calls; purely static, external links forbidden.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import httpx

SYSTEM = """You are a web developer. Build a COMPLETE, FUNCTIONAL static website in the working directory that fulfils the user's requirement.

Materialize the site so it can NEVER be "missing" (this is the single most important rule):
- Your VERY FIRST write must be index.html, and that index.html must ALREADY be a COMPLETE, self-contained, working site on its own: it must contain EVERY major feature/section the user asked for as an in-page anchor block (<section id="..."> with a heading and real content), not a thin shell that waits for other files.
- Treat additional .html files as an OPTIONAL enhancement layer added AFTER index.html is complete. If you run out of turns, a single rich index.html must still fulfil the requirement. Never leave index.html as a stub, a redirect, or an empty container to be filled by files you have not written yet.
- If you split features onto their own .html pages, index.html must STILL contain a working in-page version (or at minimum a full section with heading + real content) of each feature, so the site works even if a linked page is never reached.

Architecture (this directly affects whether features are reachable):
- You MAY also create a MULTI-PAGE site: put each major feature/section on its own .html file (index.html plus one file per feature). But index.html always comes first and always stands alone per the rule above.
- Go DEEPER than the top level: any named entity the user could inspect (a listed item, a category, a record, a match, a product, a topic, a post) must be REACHABLE — either on its own detail page or in a detail view that opens on the same page — through a real clickable element. List views must link to their detail views.
- Put an IDENTICAL navigation bar with real <a href="..."> links to EVERY page/section at the top of EVERY page, so every feature is reachable by clicking a link.
- Navigation and drill-down MUST use real hyperlinks: href to another .html file, or href="#id" pointing at a section that actually exists on that page. Do NOT rely on JavaScript alone to swap or reveal views — if a script throws, the page must still show its content.
- Every navigation link must LAND somewhere observable: the element its href targets (a #id section, or the top of the linked .html file) MUST begin with a visible <h1> or <h2> heading whose text matches that nav label. When a user clicks a nav item, they must immediately see a heading with that same wording at the landing spot — never a bare container with no heading.
- Real, human-readable content and data must already be present in the HTML before any JS runs; never leave a page or view blank waiting for JS to fill it.
- Wrap each independent widget's JavaScript in its OWN try/catch so one broken feature can never blank out the whole page.

Functionality:
- Ship a GENEROUS built-in mock dataset: at least ~8 varied, realistic records per collection, each with real fields, so searches, filters and detail views have something meaningful to show. Thin data makes working features look broken.
- Make requested features actually WORK with JS over that data: search does case-insensitive substring matching and renders the matching records into a visible results container; filters/buttons update visible content; detail views show real fields. Common, obvious queries must return visible results; show a clear empty-state only when nothing genuinely matches.
- Every <form> must call event.preventDefault() and then render a visible success/result message into an element on the page (never a bare reload, never a silent no-op).

Editable settings / account / profile / preferences forms (persist so changes survive a reload):
- For any form that lets the user EDIT and save their own data — account/contact details, profile info, user settings, preferences, a search frequency or similar option — the saved value MUST persist across page reloads using localStorage.
- On SUBMIT: read-modify-write the value(s) into localStorage (localStorage.setItem), call event.preventDefault(), and render a VISIBLE confirmation that shows the newly saved value.
- On PAGE LOAD: read the stored value(s) back from localStorage (localStorage.getItem) and populate each editable field with the stored value, so a reload shows the last saved value already filled in (not a blank/default field). This read-back is essential: it lets the saved change be observed after reload.
- Use stable, descriptive localStorage keys. Wrap this persistence logic in try/catch so a storage failure can never blank the page.

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
RE_SCRIPT = re.compile(r"<script.*?</script>", re.S | re.I)
RE_STYLE = re.compile(r"<style.*?</style>", re.S | re.I)
RE_TAG = re.compile(r"<[^>]+>")
RE_CONTROL = re.compile(r"<(input|select|textarea|button)\b", re.I)
RE_FLAT_OBJ = re.compile(r"\{[^{}]*?:[^{}]*?\}")

# anchor <a href=...>label</a> capture (label text between the tags)
RE_ANCHOR = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.S | re.I)
RE_HREF_ATTR = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.I)

# editable-field detection: text/email/number/tel/url/password inputs, selects, textareas
RE_EDITABLE = re.compile(
    r"<textarea\b|<select\b|"
    r"<input\b[^>]*type\s*=\s*['\"](?:text|email|number|tel|url|password|search|date|time)['\"]|"
    r"<input\b(?![^>]*\btype\s*=)",  # bare <input> defaults to text = editable
    re.I,
)
# settings/profile/account semantic hint (generic vocabulary, not task-specific)
RE_SETTINGS_HINT = re.compile(
    r"\b(setting|settings|preference|preferences|account|profile|"
    r"contact\s*(?:detail|info)|update|edit|frequency|configure|"
    r"my\s*info|user\s*info|personal|save\s*change)",
    re.I,
)

MAX_FIX_ROUNDS = 3
THIN_TEXT_CHARS = 150      # a linked page rendering less visible text than this = near-blank
MIN_DATA_RECORDS = 6       # search/filter present but fewer object records than this = thin data
# index.html materialization floors (the site-missing safety net):
INDEX_MIN_BYTES = 1200     # a complete self-contained index is well above this; a stub/redirect is below
INDEX_MIN_VISIBLE = 300    # visible human-readable text an index must render before any JS runs
STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "for", "my", "our",
             "your", "page", "home", "view", "go", "click", "here", "section",
             "menu", "nav", "back", "more", "all", "new", "in", "on", "at"}


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


def visible_text_len(html):
    """Approx count of human-visible text (JS-injected content NOT counted on purpose:
    the prompt requires real content in HTML before JS runs)."""
    t = RE_SCRIPT.sub(" ", html)
    t = RE_STYLE.sub(" ", t)
    t = RE_TAG.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return len(t)


def _words(s):
    """Lowercase alphanumeric tokens minus generic stopwords."""
    toks = re.findall(r"[a-z0-9]+", s.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 1}


def _headings_with_ids(html):
    """Return list of (heading_text, id_on_or_near_heading) and a set of all ids
    that carry a visible h1/h2 heading either on the element itself or containing a
    heading as an early child."""
    heading_spans = []  # (start, end, text)
    for m in re.finditer(r"<(h1|h2)\b[^>]*>(.*?)</\1>", html, re.S | re.I):
        txt = RE_TAG.sub(" ", m.group(2))
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            heading_spans.append((m.start(), m.end(), txt))
    for m in re.finditer(r'<[^>]*role\s*=\s*["\']heading["\'][^>]*>(.*?)</[^>]+>',
                         html, re.S | re.I):
        txt = RE_TAG.sub(" ", m.group(1))
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            heading_spans.append((m.start(), m.end(), txt))
    heading_spans.sort()

    id_to_heading = {}
    for m in re.finditer(r'<[^>]*\bid\s*=\s*["\']([^"\']+)["\'][^>]*>', html, re.I):
        eid = m.group(1)
        anchor_pos = m.start()
        best = ""
        for hs, he, htxt in heading_spans:
            if hs >= anchor_pos and hs - anchor_pos <= 400:
                best = htxt
                break
            if hs <= anchor_pos <= he:  # id sits on the heading tag itself
                best = htxt
                break
        if best:
            prev = id_to_heading.get(eid, "")
            id_to_heading[eid] = (prev + " " + best).strip()
    return [h[2] for h in heading_spans], id_to_heading


def _detect_edit_form(c):
    """Heuristic: does this HTML contain a settings/profile-style EDITABLE form?"""
    if not re.search(r"<form", c, re.I):
        return False
    if not RE_EDITABLE.search(c):
        return False
    return bool(RE_SETTINGS_HINT.search(c))


def analyze_site(workdir):
    """Static interaction-wiring + depth/data + persistence + nav-landing-heading audit
    + index materialization floor. Returns concrete, mechanically-found gaps.
    No requirement text is consulted -> generic."""
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

    all_html = "\n".join(html_files.values())
    has_input = bool(re.search(r"<input", all_html, re.I) or
                     re.search(r"type\s*=\s*['\"]search['\"]", all_html, re.I))
    has_filter_logic = ("filter(" in global_js or "tolowercase" in global_js or
                        "indexof(" in global_js or ".includes(" in global_js or
                        ".match(" in global_js)
    search_feature = has_input and has_filter_logic
    data_records = len(RE_FLAT_OBJ.findall("\n".join(js_blob)))
    thin_data = search_feature and data_records < MIN_DATA_RECORDS

    edit_form_present = any(_detect_edit_form(c) for c in html_files.values())
    ls_has_set = ("localstorage.setitem" in global_js or ".setitem(" in global_js)
    ls_has_get = ("localstorage.getitem" in global_js or ".getitem(" in global_js)
    persist_missing = edit_form_present and not (ls_has_set and ls_has_get)

    # --- index materialization floor (the site-missing safety net) ---
    index_exists = "index.html" in html_files
    index_bytes = 0
    index_visible = 0
    if index_exists:
        idx = html_files["index.html"]
        index_bytes = len(idx.encode("utf-8", errors="ignore"))
        index_visible = visible_text_len(idx)
    # An index counts as "materialized" only if it is present, above the byte floor,
    # AND renders real visible text before JS. A stub/redirect/empty shell fails this.
    index_under_floor = (not index_exists or
                         index_bytes < INDEX_MIN_BYTES or
                         index_visible < INDEX_MIN_VISIBLE)

    ids = {}
    heading_texts = {}
    id_headings = {}
    for rel, c in html_files.items():
        ids[rel] = set(RE_ID.findall(c)) | set(RE_NAME.findall(c))
        htexts, idh = _headings_with_ids(c)
        heading_texts[rel] = htexts
        id_headings[rel] = idh

    missing_pages, missing_anchors, unwired_forms = set(), set(), []
    nav_headingless = set()

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

    # --- nav landing-heading verification (bounded, generic) ---
    for rel, c in html_files.items():
        cdir = os.path.dirname(rel)
        for am in RE_ANCHOR.finditer(c):
            attrs, inner = am.group(1), am.group(2)
            hm = RE_HREF_ATTR.search(attrs)
            if not hm:
                continue
            h = hm.group(1).strip()
            low = h.lower()
            label = re.sub(r"\s+", " ", RE_TAG.sub(" ", inner)).strip()
            lab_words = _words(label)
            if not lab_words:
                continue  # icon-only / empty-label link: nothing to match, skip
            if low.startswith(("http://", "https://", "//", "mailto:", "tel:",
                               "javascript:", "data:")):
                continue
            frag = None
            if low.startswith("#"):
                tgt, frag = rel, h[1:]
            else:
                path = h
                if "#" in h:
                    path, frag = h.split("#", 1)
                path = path.split("?")[0]
                if path == "":
                    tgt = rel
                else:
                    base = os.path.basename(path)
                    if "." in base and not path.lower().endswith((".html", ".htm")):
                        continue
                    if path.endswith("/"):
                        path = path + "index.html"
                    tgt = os.path.normpath(os.path.join(cdir, path)).replace("\\", "/")
                    if tgt not in html_files:
                        continue  # missing page already reported elsewhere
            if frag:
                landing = id_headings.get(tgt, {}).get(frag, "")
                cand_words = _words(landing)
            else:
                cand_words = set()
                for ht in heading_texts.get(tgt, []):
                    cand_words |= _words(ht)
            if not cand_words:
                nav_headingless.add(f"{rel}: '{label}' -> {h}")
                continue
            if not (lab_words & cand_words):
                nav_headingless.add(f"{rel}: '{label}' -> {h}")

    thin_pages = []
    for rel, c in html_files.items():
        controls = len(RE_CONTROL.findall(c))
        if visible_text_len(c) < THIN_TEXT_CHARS and controls < 2:
            thin_pages.append(rel)

    return {
        "pages": sorted(html_files.keys()),
        "missing_pages": sorted(missing_pages),
        "missing_anchors": sorted(missing_anchors),
        "unwired_forms": sorted(set(unwired_forms)),
        "thin_pages": sorted(thin_pages),
        "search_feature": search_feature,
        "data_records": data_records,
        "thin_data": thin_data,
        "edit_form_present": edit_form_present,
        "persist_missing": persist_missing,
        "nav_headingless": sorted(nav_headingless),
        "index_exists": index_exists,
        "index_bytes": index_bytes,
        "index_visible": index_visible,
        "index_under_floor": index_under_floor,
    }


def build_fix_message(rep):
    lines = ["Not done yet - a static navigation/wiring/completeness check found problems. "
             "Fix ALL of these, then output <done/> again:"]
    if rep["index_under_floor"]:
        if not rep["index_exists"]:
            lines.append("- index.html does NOT exist on disk, so the whole site is unopenable and "
                         "every test fails. Your FIRST action must be to write a COMPLETE, "
                         "self-contained index.html that already contains EVERY requested feature as "
                         "an in-page <section id=\"...\"> block with a heading and real content, plus "
                         "the shared nav bar. Do not depend on any other file.")
        else:
            lines.append("- index.html exists but is only %d bytes / %d visible characters — it looks "
                         "like a stub, redirect, or empty shell rather than a complete site, so when "
                         "the site is opened it appears missing/blank. REWRITE index.html as a COMPLETE, "
                         "self-contained page that already contains EVERY requested feature as an "
                         "in-page <section id=\"...\"> block with a visible heading and real, "
                         "human-readable content, plus the shared nav bar, so the site fully works from "
                         "index.html ALONE even if no other file is reached."
                         % (rep["index_bytes"], rep["index_visible"]))
    if rep["missing_pages"]:
        lines.append("- These pages are LINKED but do not exist. Create each one as a real, "
                     "functional page with the same shared nav bar and real content: "
                     + ", ".join(rep["missing_pages"]))
    if rep["missing_anchors"]:
        lines.append("- These links point to sections/ids that DO NOT EXIST (dead nav -> blank view). "
                     "Either add an element with that id, or fix the href: "
                     + ", ".join(rep["missing_anchors"]))
    if rep["nav_headingless"]:
        lines.append("- These navigation links LAND somewhere with no visible heading that matches "
                     "the label the user clicked, so clicking them looks like nothing happened. "
                     "For EACH: make the target (the #id section, or the top of the linked .html "
                     "file) BEGIN with a visible <h1> or <h2> whose text matches the link label, so "
                     "the user immediately sees a heading with that same wording after clicking: "
                     + ", ".join(rep["nav_headingless"][:20]))
    if rep["thin_pages"]:
        lines.append("- These pages exist but render almost NO visible content before JS runs "
                     "(they look blank/broken when reached). Build them out with real, human-readable "
                     "content and data plus the shared nav bar, so the feature actually works: "
                     + ", ".join(rep["thin_pages"]))
    if rep["thin_data"]:
        lines.append("- Search/filter logic is present but only ~%d built-in data records were detected, "
                     "so common queries will return NOTHING. Expand the built-in mock dataset to at least "
                     "8-12 varied, realistic records (objects with real fields) and make search do "
                     "case-insensitive substring matching that renders matches into a visible container."
                     % rep["data_records"])
    if rep["persist_missing"]:
        lines.append("- An editable settings/account/profile form is present, but the saved value will "
                     "NOT survive a page reload (no localStorage read+write detected). For each such "
                     "form: on submit call event.preventDefault(), write the field value(s) with "
                     "localStorage.setItem(...) and show a visible confirmation of the saved value; AND "
                     "on page load read the stored value(s) with localStorage.getItem(...) and populate "
                     "each editable field with it, so after a reload the last saved value is already "
                     "shown in the field. Wrap this persistence logic in try/catch.")
    if rep["unwired_forms"]:
        lines.append("- These pages contain a <form> with no submit handler anywhere. Add an onsubmit "
                     "handler that calls event.preventDefault() and shows a visible result message: "
                     + ", ".join(rep["unwired_forms"]))
    return "\n".join(lines)


def has_blocking_gap(rep):
    return (rep["index_under_floor"] or rep["missing_pages"]
            or rep["missing_anchors"] or rep["unwired_forms"]
            or rep["thin_pages"] or rep["thin_data"]
            or rep["persist_missing"] or rep["nav_headingless"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-json", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model", default="Qwen3.5-9B")
    ap.add_argument("--base-url", default="https://api.example.com/v1")
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    t0 = time.time()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    key = Path(args.api_key_file).read_text().strip()
    task = json.loads(Path(args.task_json).read_text())

    user0 = (f"Requirement:\n{task['instruction']}\n\n"
             f"Application type: {task.get('application_type', 'website')}\n\n"
             "Build the full site now. Your FIRST write must be a COMPLETE, self-contained "
             "index.html that already contains every requested feature as an in-page section "
             "with the shared navigation bar; add extra pages only afterwards as enhancement.")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user0}]

    calls = parse_misses = llm_errors = link_fix_rounds = index_floor_fix_rounds = 0
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
                if rep["index_under_floor"]:
                    index_floor_fix_rounds += 1
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
        "persist_check_ran": True,
        "heading_check_ran": True,
        "index_floor_check_ran": True,
        "index_bytes_final": final["index_bytes"],
        "index_visible_final": final["index_visible"],
        "index_under_floor_final": final["index_under_floor"],
        "index_floor_fix_rounds": index_floor_fix_rounds,
        "pages_generated": len(final["pages"]),
        "missing_links_final": missing_final,
        "unwired_forms_final": len(final["unwired_forms"]),
        "thin_pages_final": len(final["thin_pages"]),
        "data_records_final": final["data_records"],
        "search_feature": final["search_feature"],
        "thin_data_final": final["thin_data"],
        "edit_form_present": final["edit_form_present"],
        "persist_missing": final["persist_missing"],
        "nav_headingless_final": len(final["nav_headingless"]),
        "link_fix_rounds": link_fix_rounds,
    }, indent=1))


if __name__ == "__main__":
    main()
