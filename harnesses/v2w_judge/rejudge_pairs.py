#!/usr/bin/env python3
"""Re-judge harness: render page pairs (3 viewports + DOM innerText) and run a judge.

Inputs:
  --pairs   pairs.json: {"pairs": [{"id", "task", "seed",
                                     "A": {"id", "dir"}, "B": {"id", "dir"},
                                     "label_A", "label_B"}, ...]}
  --judge   path to a judge script with the jgen2 CLI contract
            (--exam --renders --datasets --out --workdir --concurrency)
  --renders output render cache dir (created; per-candidate <id>/<vp>.png + dom.txt)
  --out     verdicts.jsonl path
  --workdir scratch dir (exam.json + judge_work)

Rendering mirrors phase2_bon_pipeline/render_cache_pages.py (local http.server per page,
external requests aborted, full-page screenshots) and additionally dumps
document.body.innerText (desktop viewport) to <renders>/<id>/dom.txt.
Resumable: candidates with all 3 pngs + dom.txt are skipped; judges skip ids already
present in the verdicts file.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

VPS = {"desktop": 1920, "tablet": 1024, "mobile": 375}


def wait_port(port, sec=8):
    t0 = time.time()
    while time.time() - t0 < sec:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return True
        except OSError:
            time.sleep(0.2)
    return False


def render_pages(pages, renders_dir, port):
    from playwright.sync_api import sync_playwright

    renders_dir = Path(renders_dir)
    renders_dir.mkdir(parents=True, exist_ok=True)
    done = fail = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        for pid, pdir in sorted(pages.items()):
            out = renders_dir / pid
            if all((out / f"{vp}.png").exists() for vp in VPS) and (out / "dom.txt").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            srv = subprocess.Popen(
                ["python3", "-m", "http.server", str(port), "--directory", pdir,
                 "--bind", "127.0.0.1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                if not wait_port(port):
                    raise RuntimeError("no server")
                for vp, w in VPS.items():
                    ctx = browser.new_context(viewport={"width": w, "height": 1100})
                    pg = ctx.new_page()
                    pg.route("**/*", lambda rt: rt.continue_()
                             if rt.request.url.startswith(f"http://127.0.0.1:{port}")
                             else rt.abort())
                    try:
                        pg.goto(f"http://127.0.0.1:{port}/index.html",
                                wait_until="domcontentloaded", timeout=15000)
                        pg.wait_for_timeout(800)
                        pg.screenshot(path=str(out / f"{vp}.png"), full_page=True, timeout=20000)
                        if vp == "desktop":
                            txt = pg.evaluate("() => document.body ? document.body.innerText : ''")
                            (out / "dom.txt").write_text(txt or "")
                    finally:
                        ctx.close()
                done += 1
            except Exception as e:
                fail += 1
                print(f"FAIL {pid}: {str(e)[:80]}", flush=True)
            finally:
                srv.terminate()
                try:
                    srv.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    srv.kill()
        browser.close()
    print(f"render done={done} fail={fail}", flush=True)
    return fail


def run_judge(exam, judge, renders_dir, datasets, out, workdir, concurrency):
    """Drive a jgen2-contract judge over `exam` (list of {id, task, A, B}).

    Writes/append-resumes verdicts at `out`; scratch under `workdir`.
    Returns {pair_id: verdict_record}.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    exam_path = workdir / "exam.json"
    exam_path.write_text(json.dumps(exam, indent=2) + "\n")

    env = dict(os.environ)
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    subprocess.run(
        [sys.executable, str(judge),
         "--exam", str(exam_path),
         "--renders", str(renders_dir),
         "--datasets", str(datasets),
         "--out", str(out),
         "--workdir", str(workdir / "judge_work"),
         "--concurrency", str(concurrency)],
        check=True, env=env)

    verdicts = {}
    out_path = Path(out)
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            rec = json.loads(line)
            verdicts[rec["id"]] = rec
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--datasets",
                    default="./datasets/webpage")
    ap.add_argument("--port", type=int, default=26410)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs).read_text())["pairs"]
    renders_dir = Path(args.renders)
    workdir = Path(args.workdir)
    renders_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    pages = {}
    for pair in pairs:
        for side in ("A", "B"):
            cand = pair[side]
            if not (Path(cand["dir"]) / "index.html").exists():
                raise SystemExit(f"missing index.html: {cand['dir']}")
            pages[cand["id"]] = cand["dir"]

    fail = render_pages(pages, renders_dir, args.port)
    missing = [pid for pid in pages
               if not all((renders_dir / pid / f"{vp}.png").exists() for vp in VPS)]
    if missing:
        raise SystemExit(f"incomplete renders: {missing}")
    if args.render_only:
        print("render-only mode; skipping judge")
        return

    exam = [{"id": pair["id"], "task": pair["task"],
             "A": pair["A"]["id"], "B": pair["B"]["id"]} for pair in pairs]
    verdicts = run_judge(exam, args.judge, renders_dir, args.datasets,
                         args.out, workdir, args.concurrency)
    missing_verdicts = [p["id"] for p in pairs if p["id"] not in verdicts]
    if missing_verdicts:
        raise SystemExit(f"missing verdicts: {missing_verdicts}")
    print(json.dumps({"judged": len(verdicts), "out": args.out}))


if __name__ == "__main__":
    main()
