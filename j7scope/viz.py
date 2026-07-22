"""Bilingual side-by-side readout views.

Currently a static HTML table: one row per zh/en prompt pair with both sides'
top-k J-lens readouts and a concept-level overlap score. TODO: port the
upstream slice-vis d3 page (anthropics/jacobian-lens) to a two-column
bilingual layout for interactive exploration.
"""

import html
from pathlib import Path

_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>J7Scope — zh/en J-lens readouts</title>
<style>
  body {{ font: 14px/1.5 -apple-system, "Segoe UI", sans-serif; margin: 2rem; color: #222; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: .5rem .7rem; vertical-align: top; text-align: left; }}
  th {{ background: #f5f5f5; }}
  .prompt {{ color: #555; font-size: 13px; }}
  .tok {{ display: inline-block; background: #eef3fa; border-radius: 4px;
          padding: 0 .35em; margin: .1em .15em; font-family: ui-monospace, monospace; }}
  .overlap {{ font-weight: 600; text-align: center; }}
</style>
<h1>J7Scope — bilingual J-lens readouts</h1>
<table>
<tr><th>id / concept</th><th>zh</th><th>en</th><th>overlap</th></tr>
{rows}
</table>
"""

_ROW = """<tr>
<td><b>{id}</b><br>{concept}</td>
<td><div class="prompt">{zh_text}</div>{zh_toks}</td>
<td><div class="prompt">{en_text}</div>{en_toks}</td>
<td class="overlap">{overlap}</td>
</tr>"""


def _tokens(topk):
    return "".join(f'<span class="tok" title="{v}">{html.escape(str(t))}</span>'
                   for t, v in topk)


def render_comparison_html(rows, path):
    """Write a standalone comparison page.

    rows: iterable of dicts with keys id, concept, zh_text, en_text,
          zh_topk, en_topk ([(token, logit), ...]) and overlap (float).
    """
    body = "\n".join(_ROW.format(
        id=html.escape(str(r["id"])),
        concept=html.escape(str(r["concept"])),
        zh_text=html.escape(r["zh_text"]),
        en_text=html.escape(r["en_text"]),
        zh_toks=_tokens(r["zh_topk"]),
        en_toks=_tokens(r["en_topk"]),
        overlap=f'{r["overlap"]:.2f}',
    ) for r in rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PAGE.format(rows=body), encoding="utf-8")
    return path
