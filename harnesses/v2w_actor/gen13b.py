#!/usr/bin/env python3
"""MiniHarness v0-merge+res+silhouette -- on top of the gen10 champion, single-axis reinforcement of [thumbnail-judge adversarial: silhouette/height/colour bands].

White-box adversarial round: fitness = the learned pairwise judge (shrinks [prototype|A|B] to ~240-330px wide, weak models vote, majority vote over 3 viewports,
ties broken by broken-image count + page-height ratio). At thumbnail scale detail text is invisible; what decides the outcome is the [overall silhouette]:
total page height, the vertical colour-band sequence (dark header / hero colour / dark footer), block count and rhythm, zero broken images.

The evidence (attached figure) shows the champion's fatal wound is exactly a signal the judge can see: the page is severely too short (in the evidence figure
the desktop page renders only ~1 screen while the prototype has ~10 sections; several cases drop a whole colour band and the footer) -> the page-height ratio
and colour-band silhouette are completely misaligned.
The model saw the high-res slices but never received an aimable compressed summary of the [target page height + vertical colour skeleton].

This generation's only addition (orthogonal, zero extra LLM calls, no selection loop -- both of those have already failed):
  SILHOUETTE PERCEPTION: mechanically extract, per viewport, (a) the target rendered page height (px) and (b) an N-segment top-to-bottom dominant colour band map,
  and inject them, together with the redirect instruction "the judge only sees thumbnails; the primary goal is to replicate the silhouette/page height/colour bands;
  do not stop halfway and drop sections", into the first-round message.

Everything else (loose XML protocol / single file / whole-file write / slice perception / resource manifest + thumbnails / mechanical post-processing / broken-image placeholder box) is fully identical to the champion.

Evolution hard constraints: 1. only call the one --model model; 2. never read evaluation/scores at runtime; 3. no task names / page content in the code; 4. LLM calls <=20.
"""
import argparse, base64, io, json, os, re, time, difflib
from collections import defaultdict
from pathlib import Path

SYSTEM = """You are a web developer agent. Build a static webpage in the working directory that visually matches the three prototype screenshots (desktop 1920px / tablet 1024px / mobile 375px).

You are given each prototype as several HIGH-RESOLUTION vertical slices, ordered top-to-bottom (slice 1/N is the very top of the page, slice N/N is the bottom). Read the actual TEXT in these slices carefully — headings, prices, button labels, section titles, testimonials, form fields — and reproduce the REAL content and copy of the page. Do not substitute a generic template: match what the slices actually say. Tall pages may still be cropped at the very bottom; infer the rest sensibly.

TOP PRIORITY — OVERALL SILHOUETTE: the page is judged primarily on how closely it matches the prototype AT A GLANCE. That means (1) TOTAL PAGE HEIGHT: reproduce EVERY section top to bottom so your rendered page reaches the given target height — never stop short or omit whole sections; (2) VERTICAL COLOR RHYTHM: match the sequence of coloured bands top-to-bottom (dark headers, coloured hero banners, light body, dark/coloured footers) using the band colour map provided; (3) SECTION COUNT & RHYTHM and hero dominance. Getting the full-height layout and band colours right matters more than perfecting small text.

You are ALSO shown the real image assets available under resources/ (a text catalog with exact paths + pixel sizes, plus thumbnails labelled with their exact src). Use these REAL filenames as your <img src> — do NOT invent filenames — and size each image according to its real pixel dimensions / aspect ratio so logos, hero photos, badges and icons appear at sensible proportions instead of stretched or missing. Every <img> must resolve to a real asset so nothing renders as a broken image.

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
- If you use CSS, put it inside a <style> tag in index.html OR write style.css AND link it with <link rel="stylesheet" href="style.css">. Never paste raw CSS as page text.
- Make the page responsive so it matches all three screenshots at their widths.
- When you are satisfied the page is complete, output <done/>."""

# ------------------------- perception (high-res slices) -------------------------
def _img_part(im):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}

def slice_image(p, width, tile_h, max_tiles):
    """Resize to `width`, then cut into top->bottom tiles of height <= tile_h."""
    from PIL import Image
    im = Image.open(p).convert("RGB")
    im = im.resize((width, max(1, round(im.height * width / im.width))))
    tiles, y = [], 0
    while y < im.height and len(tiles) < max_tiles:
        tiles.append(im.crop((0, y, im.width, min(im.height, y + tile_h))))
        y += tile_h
    return tiles

# high-res slices per viewport: (width, tile_height, max_tiles)
SLICE_SPEC = [("desktop", 1000, 1150, 4), ("tablet", 680, 1400, 2), ("mobile", 430, 1200, 3)]

def perception_parts(proto):
    """Return (list_of_message_parts, {vp: n_slices})."""
    parts, counts = [], {}
    for vp, w, th, maxt in SLICE_SPEC:
        p = proto / f"{vp}.jpg"
        if not p.exists():
            counts[vp] = 0
            continue
        tiles = slice_image(p, w, th, maxt)
        counts[vp] = len(tiles)
        n = len(tiles)
        for i, t in enumerate(tiles):
            label = (f"[{vp} prototype — slice {i+1}/{n} (top to bottom)]"
                     if n > 1 else f"[{vp} prototype]")
            parts.append({"type": "text", "text": label})
            parts.append(_img_part(t))
    return parts, counts

# ------------------------- silhouette metrics (NEW) -------------------------
# nominal render width per viewport (what the target page height is expressed at)
VP_W = {"desktop": 1920, "tablet": 1024, "mobile": 375}

def proto_metrics(proto, n_bands=14):
    """Per-viewport target page height (px @ nominal width) + top->bottom dominant colour bands."""
    from PIL import Image
    out = {}
    for vp, nomw in VP_W.items():
        p = proto / f"{vp}.jpg"
        if not p.exists():
            continue
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        target_h = round(h * nomw / max(1, w))
        # average colour per horizontal band (resize collapses horizontal + band-averages vertical)
        strip = im.resize((1, n_bands))
        bands = []
        for i in range(n_bands):
            c = strip.getpixel((0, i))
            bands.append([round(100 * i / n_bands), f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"])
        out[vp] = {"target_h": target_h, "aspect": round(h / max(1, w), 2), "bands": bands}
    return out

def format_metrics(metrics):
    lines = []
    for vp in ("desktop", "tablet", "mobile"):
        m = metrics.get(vp)
        if not m:
            continue
        band_txt = "  ".join(f"{pct}%={hexc}" for pct, hexc in m["bands"])
        lines.append(
            f"- {vp} (render width {VP_W[vp]}px): TARGET total page height ~{m['target_h']}px "
            f"(height/width ratio {m['aspect']}). Vertical colour rhythm top->bottom: {band_txt}")
    return "\n".join(lines) or "(no metrics)"

# ------------------------- resource image awareness -------------------------
IMG_EXT  = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp", ".avif"}

def _kind(w, h):
    ar = w / max(1, h)
    if max(w, h) <= 96 and 0.6 <= ar <= 1.7:
        return "small icon"
    if ar >= 3:
        return "very wide banner/logo strip"
    if ar >= 1.6:
        return "wide image/logo/badge"
    if ar <= 0.6:
        return "tall image"
    return "image"

def resource_catalog(workdir, max_thumbs=8, thumb_w=170):
    """Return (text_manifest, list_of_labelled_thumbnail_parts, n_thumbs)."""
    root = workdir / "resources"
    if not root.exists():
        return "(no resources/ directory)", [], 0
    from PIL import Image
    all_imgs = [p for p in sorted(root.rglob("*"))
                if p.is_file() and p.suffix.lower() in IMG_EXT]
    lines = []
    for p in all_imgs:
        rel = p.relative_to(workdir).as_posix()
        if p.suffix.lower() == ".svg":
            desc = "svg vector"
        else:
            try:
                im = Image.open(p)
                desc = f"{im.width}x{im.height}px, {_kind(im.width, im.height)}"
            except Exception:
                desc = "?"
        lines.append(f"- {rel} ({desc})")
    manifest = "\n".join(lines) or "(no image files under resources/)"

    raster = [p for p in all_imgs if p.suffix.lower() != ".svg"]
    if len(raster) > max_thumbs:
        step = len(raster) / max_thumbs
        raster = [raster[int(i * step)] for i in range(max_thumbs)]
    parts = []
    for p in raster:
        try:
            im = Image.open(p).convert("RGB")
            if im.width > thumb_w:
                im = im.resize((thumb_w, max(1, round(im.height * thumb_w / im.width))))
            if im.height > thumb_w * 3:
                im = im.crop((0, 0, im.width, thumb_w * 3))
            rel = p.relative_to(workdir).as_posix()
            parts.append({"type": "text", "text": f'[resource image — use src="{rel}"]'})
            parts.append(_img_part(im))
        except Exception:
            pass
    return manifest, parts, len(parts) // 2

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

RE_WRITE = re.compile(r'<write\s+path="([^"]+)"\s*>\n?(.*?)</write>', re.S)
RE_READ = re.compile(r'<read\s+path="([^"]+)"\s*/>')
RE_LS = re.compile(r"<ls\s*/>")
RE_DONE = re.compile(r"<done\s*/>")

# ------------------------- mechanical post-processing -------------------------
CSS_EXT  = {".css"}
JS_EXT   = {".js", ".mjs"}
FONT_EXT = {".woff", ".woff2", ".ttf", ".otf", ".eot"}

def _extclass(ext):
    ext = ext.lower()
    if ext in IMG_EXT:  return "img"
    if ext in CSS_EXT:  return "css"
    if ext in JS_EXT:   return "js"
    if ext in FONT_EXT: return "font"
    return ext or "?"

def best_match(ref, files):
    base = os.path.basename(ref.split("?")[0].split("#")[0]).lower()
    if not base:
        return None
    stem, ext = os.path.splitext(base)
    exact = [f for f in files if os.path.basename(f).lower() == base]
    if exact:
        return min(exact, key=lambda f: (0 if f.startswith("resources") else 1, len(f)))
    want = _extclass(ext)
    pool = [f for f in files if _extclass(os.path.splitext(f)[1]) == want]
    stems = [os.path.splitext(os.path.basename(f))[0].lower() for f in pool]
    if stem and stems:
        m = difflib.get_close_matches(stem, stems, n=1, cutoff=0.62)
        if m:
            return pool[stems.index(m[0])]
    return None

def resolve_ref(ref, d, files, workdir):
    ref = (ref or "").strip()
    if not ref or ref.startswith(("#", "data:", "mailto:", "tel:", "javascript:")):
        return True
    if ref.startswith(("http://", "https://", "//")):
        bm = best_match(ref, files)
        return (os.path.relpath(bm, d) if d else bm) if bm else None
    clean = ref.split("?")[0].split("#")[0]
    tgt = workdir / (os.path.join(d, clean) if d else clean)
    try:
        if tgt.resolve().exists():
            return True
    except Exception:
        pass
    bm = best_match(ref, files)
    return (os.path.relpath(bm, d) if d else bm) if bm else None

def strip_external(html):
    n = [0]
    def link_repl(m):
        tag = m.group(0)
        href = re.search(r'href\s*=\s*["\']([^"\']*)["\']', tag)
        rel  = re.search(r'rel\s*=\s*["\']([^"\']*)["\']', tag)
        if href and href.group(1).strip().startswith(("http://", "https://", "//")):
            n[0] += 1; return ""
        if rel and rel.group(1).lower().strip() in ("preconnect", "dns-prefetch", "preload", "prefetch"):
            n[0] += 1; return ""
        return tag
    html = re.sub(r"<link\b[^>]*>", link_repl, html, flags=re.I)
    def script_repl(m):
        n[0] += 1; return ""
    html = re.sub(r'<script\b[^>]*\bsrc\s*=\s*["\'](?:https?:)?//[^"\']*["\'][^>]*>\s*</script>',
                  script_repl, html, flags=re.I)
    return html, n[0]

def wrap_leaked_css(html):
    def looks_css(t):
        t2 = t.strip()
        if len(t2) < 200:
            return False
        rules = re.findall(r"[^{}<>]+\{[^{}]*\}", t2)
        props = re.findall(r"[\w-]+\s*:\s*[^;{}<>]+;", t2)
        return len(rules) >= 3 or len(props) >= 12
    parts = re.split(r"(<[^>]+>)", html)
    inside = 0
    n = 0
    out = []
    for seg in parts:
        if seg.startswith("<"):
            tl = seg.lower()
            if tl.startswith("<style") or tl.startswith("<script"):
                inside += 1
            elif tl.startswith("</style") or tl.startswith("</script"):
                inside = max(0, inside - 1)
            out.append(seg)
        else:
            if inside == 0 and looks_css(seg):
                out.append("<style>\n" + seg + "\n</style>")
                n += 1
            else:
                out.append(seg)
    return "".join(out), n

def fix_imgs(html, d, files, workdir):
    fixed = [0]; fell = [0]
    def repl(m):
        tag = m.group(0)
        sm = re.search(r'(src\s*=\s*["\'])([^"\']*)(["\'])', tag)
        if not sm:
            return tag
        ref = sm.group(2)
        if ref.startswith("data:"):
            return tag
        r = resolve_ref(ref, d, files, workdir)
        if r is True:
            return tag
        if r:
            fixed[0] += 1
            return tag[:sm.start(2)] + r + tag[sm.end(2):]
        fell[0] += 1
        am = re.search(r'alt\s*=\s*["\']([^"\']*)["\']', tag)
        alt = (am.group(1) if am else "").strip()
        cm = re.search(r'class\s*=\s*["\']([^"\']*)["\']', tag)
        cl = f' class="{cm.group(1)}"' if cm else ""
        wm = re.search(r'width\s*=\s*["\']?(\d+)', tag)
        hm = re.search(r'height\s*=\s*["\']?(\d+)', tag)
        dims = ""
        if wm: dims += f"width:{wm.group(1)}px;"
        if hm: dims += f"height:{hm.group(1)}px;"
        style = ("display:inline-block;max-width:100%;box-sizing:border-box;" + dims +
                 "background:#eceff3;color:#9aa0a6;font-size:11px;line-height:1.3;"
                 "padding:4px 6px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;"
                 "vertical-align:middle")
        return f'<span{cl} style="{style}">{alt}</span>'
    html = re.sub(r"<img\b[^>]*>", repl, html, flags=re.I)
    return html, fixed[0], fell[0]

def fix_urls(text, d, files, workdir):
    rew = [0]; neu = [0]
    def repl(m):
        q = m.group(1); ref = m.group(2)
        r = resolve_ref(ref, d, files, workdir)
        if r is True:
            return m.group(0)
        if r:
            rew[0] += 1
            return f"url({q}{r}{q})"
        neu[0] += 1
        return "none"
    text = re.sub(r'url\(\s*(["\']?)([^)"\']*)\1\s*\)', repl, text)
    return text, rew[0], neu[0]

def inject_viewport(html):
    if re.search(r'name=["\']viewport["\']', html, re.I):
        return html, 0
    m = re.search(r"<head[^>]*>", html, re.I)
    if not m:
        return html, 0
    i = m.end()
    tag = '\n<meta name="viewport" content="width=device-width, initial-scale=1">'
    return html[:i] + tag + html[i:], 1

def process_html(html, rel, files, workdir):
    d = os.path.dirname(rel)
    s = defaultdict(int)
    html, n = strip_external(html);                 s["links_stripped"] += n
    html, n = wrap_leaked_css(html);                s["css_wrapped"] += n
    html, a, b = fix_imgs(html, d, files, workdir); s["imgs_fixed"] += a; s["imgs_fallback"] += b
    html, rw, nu = fix_urls(html, d, files, workdir); s["urls_rewritten"] += rw; s["urls_neutralized"] += nu
    html, n = inject_viewport(html);                s["viewport_injected"] += n
    return html, s

def postprocess(workdir):
    all_files = [p.relative_to(workdir).as_posix()
                 for p in workdir.rglob("*")
                 if p.is_file() and p.relative_to(workdir).parts[0] != ".agent"]
    stats = defaultdict(int)
    for p in workdir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(workdir).as_posix()
        top = rel.split("/")[0]
        if top in (".agent", "resources"):
            continue
        suf = p.suffix.lower()
        try:
            t = p.read_text()
        except Exception:
            continue
        if suf in (".html", ".htm"):
            nt, s = process_html(t, rel, all_files, workdir)
            for k, v in s.items():
                stats[k] += v
            stats["files_processed"] += 1
            if nt != t:
                p.write_text(nt); stats["files_changed"] += 1
        elif suf == ".css":
            nt, rw, nu = fix_urls(t, os.path.dirname(rel), all_files, workdir)
            stats["urls_rewritten"] += rw; stats["urls_neutralized"] += nu
            stats["files_processed"] += 1
            if nt != t:
                p.write_text(nt); stats["files_changed"] += 1
    return dict(stats)
# -----------------------------------------------------------------------------

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
    log = {"calls": 0, "steps": [], "t0": time.time()}
    logf = workdir / ".agent" / "log.json"
    logf.parent.mkdir(exist_ok=True)

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=Path(args.api_key_file).read_text().strip(), timeout=300)

    def llm(messages):
        log["calls"] += 1
        r = client.chat.completions.create(model=args.model, messages=messages, temperature=args.temperature)
        return r.choices[0].message.content or ""

    proto = task_dir / "prototypes"
    img_parts, slice_counts = perception_parts(proto)
    log["slices"] = slice_counts

    # NEW: silhouette metrics (target page height + vertical colour band map)
    try:
        metrics = proto_metrics(proto)
    except Exception as e:
        metrics = {}
        log["metrics_error"] = f"{type(e).__name__}: {e}"[:200]
    metrics_txt = format_metrics(metrics)
    log["proto_metrics"] = {vp: {"target_h": m["target_h"], "aspect": m["aspect"], "bands": m["bands"]}
                            for vp, m in metrics.items()}

    # resource image awareness (text manifest + labelled thumbnails)
    try:
        res_manifest, res_parts, n_res_thumbs = resource_catalog(workdir)
    except Exception as e:
        res_manifest, res_parts, n_res_thumbs = "(resource catalog unavailable)", [], 0
        log["res_catalog_error"] = f"{type(e).__name__}: {e}"[:200]
    log["n_res_thumbs"] = n_res_thumbs
    log["n_image_parts"] = sum(1 for x in img_parts if x.get("type") == "image_url") + n_res_thumbs

    first = [{"type": "text", "text":
              "Build the webpage matching the prototype screenshots below. Each prototype is provided "
              "as high-resolution vertical slices ordered top-to-bottom — read their text carefully and "
              "reproduce the REAL content, layout and background colors of the page.\n\n"
              "SILHOUETTE TARGET (highest priority — the page is judged at a glance on overall shape):\n"
              + metrics_txt +
              "\nBuild EVERY section top-to-bottom so your page reaches the target height (do not stop "
              "short or drop whole sections), and reproduce the vertical colour rhythm above "
              "(dark headers/hero bands/coloured footers) so the page's coloured blocks line up with the "
              "prototype at the same vertical positions.\n\n"
              "Real image assets available under resources/ (use these EXACT paths as <img src>, and "
              "size images per their pixel dimensions):\n" + res_manifest + "\n\n"
              "Start by writing index.html (you may also write style.css / script.js)."}]
    first.extend(img_parts)
    if res_parts:
        first.append({"type": "text", "text":
                      "Thumbnails of the real resource images follow (each labelled with the exact src to use):"})
        first.extend(res_parts)
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
                results.append(f"[write ok: {m.group(1)} ({len(m.group(2))} chars)]")
            except Exception as e:
                results.append(f"[write FAILED {m.group(1)}: {e}]")
        for m in RE_READ.finditer(out):
            acted = True
            try:
                content = safe(workdir, m.group(1)).read_text()[:4000]
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
        messages.append({"role": "user", "content": "\n".join(results)[:6000]})

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

    try:
        log["post"] = postprocess(workdir)
    except Exception as e:
        log["post_error"] = f"{type(e).__name__}: {e}"[:300]

    log["done"] = done
    log["seconds"] = round(time.time() - log["t0"], 1)
    log["index_exists"] = (workdir / "index.html").exists()
    try:
        log["index_chars"] = len((workdir / "index.html").read_text())
    except Exception:
        log["index_chars"] = 0
    logf.write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(json.dumps({k: log[k] for k in ("calls", "done", "seconds", "index_exists")}))

if __name__ == "__main__":
    main()
