#!/usr/bin/env python3
"""Export a run's result.json into a readable speaker-attributed transcript.

Reads the FinalSegment list produced by scripts/run_episode.py (or main.py)
and renders:

  * a plain-text transcript (``<id>.txt``)   — for quick reading/printing,
  * an HTML transcript    (``<id>.html``)    — styled, for sharing/defense.

Usage:
    python scripts/export_transcript.py data/output/rtv_goll_table_ep/result.json
    python scripts/export_transcript.py PATH/TO/result.json --out my_transcript

Outputs are written next to the input by default (``<stem>.txt/.html``), or to
``--out`` with ``--out`` treated as a directory when it is one.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

# Repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_segments(result_json: Path) -> list[dict]:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{result_json} does not contain a list of segments")
    return data


def _fmt_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}".replace(".", ".") if h else f"{m:02d}:{s:05.2f}"


def render_text(segments: list[dict]) -> str:
    lines = []
    for s in segments:
        speaker = s.get("speaker", "?")
        text = (s.get("text") or "").strip()
        conf = s.get("confidence")
        conf_s = f"  [conf={conf:.2f}]" if isinstance(conf, (int, float)) else ""
        lines.append(f"[{_fmt_time(s.get('start', 0))} - {_fmt_time(s.get('end', 0))}] {speaker}{conf_s}")
        lines.append(f"    {text}")
        lines.append("")
    return "\n".join(lines)


def render_html(segments: list[dict], title: str = "Speaker-Attributed Transcript") -> str:
    parts = [
        "<!DOCTYPE html>",
        '<html lang="bn"><head><meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Georgia,'Times New Roman',serif;max-width:900px;margin:2rem auto;"
        "padding:0 1rem;line-height:1.55;color:#1a1a1a}",
        "h1{font-size:1.4rem;border-bottom:2px solid #1a3c6e;padding-bottom:.3rem}",
        ".seg{margin:1.1rem 0;padding:.7rem .9rem;border-left:4px solid #1a3c6e;"
        "background:#f7f9fc;border-radius:0 6px 6px 0}",
        ".meta{font-size:.8rem;color:#5a6b7d;font-family:Menlo,Consolas,monospace}",
        ".spk{font-weight:bold;color:#1a3c6e}",
        ".text{margin-top:.3rem;font-size:1.05rem}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    for s in segments:
        speaker = html.escape(str(s.get("speaker", "?")))
        text = html.escape((s.get("text") or "").strip()) or "…"
        conf = s.get("confidence")
        conf_s = f" · conf={conf:.2f}" if isinstance(conf, (int, float)) else ""
        t0, t1 = _fmt_time(s.get("start", 0)), _fmt_time(s.get("end", 0))
        parts.append(
            f'<div class="seg"><div class="meta">[{t0} – {t1}]'
            f'<span class="spk"> {speaker}</span>{conf_s}</div>'
            f'<div class="text">{text}</div></div>'
        )
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("result_json", help="Path to result.json")
    ap.add_argument("--out", default=None,
                    help="Output path (or directory). Default: next to input, "
                         "named <stem>.txt and <stem>.html")
    ap.add_argument("--title", default=None, help="Title for the HTML page")
    args = ap.parse_args()

    src = Path(args.result_json).resolve()
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 2

    segments = load_segments(src)
    if args.out:
        out_target = Path(args.out).resolve()
        if out_target.exists() and out_target.is_dir():
            out_dir = out_target
            base = src.stem
        else:
            out_dir = out_target.parent
            base = out_target.stem
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = src.parent
        base = src.stem

    txt_path = out_dir / f"{base}.txt"
    html_path = out_dir / f"{base}.html"

    txt_path.write_text(render_text(segments), encoding="utf-8")
    html_path.write_text(render_html(segments, args.title or f"Transcript — {base}"),
                         encoding="utf-8")

    print(f"✔ text : {txt_path}")
    print(f"✔ html : {html_path}")
    print(f"  segments: {len(segments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
