from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def write_demo_html(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write a self-contained interactive stale-memory demo page."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(demo_html(payload), encoding="utf-8")
    return path


def demo_html(payload: dict[str, Any]) -> str:
    data = _json_for_script(payload)
    title = escape(str(payload.get("case_id") or "AdaMem demo"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AdaMem State Authority Demo - {title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --panel: #eef2f6;
      --ink: #15202b;
      --muted: #5b6776;
      --line: #d7dee8;
      --teal: #0f766e;
      --blue: #2563eb;
      --amber: #b45309;
      --red: #b91c1c;
      --green: #15803d;
      --shadow: 0 10px 28px rgba(21, 32, 43, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      min-width: 320px;
    }}
    .shell {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr;
    }}
    header {{
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      padding: 16px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      font-weight: 720;
      letter-spacing: 0;
    }}
    .subhead {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }}
    .status-strip {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--surface);
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .chip.good {{ color: var(--green); border-color: #b7dfc2; background: #f0fbf3; }}
    .chip.warn {{ color: var(--amber); border-color: #f0d8a8; background: #fff8ec; }}
    .summary {{
      padding: 12px 22px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .boundary {{
      padding: 0 22px 12px;
    }}
    .summary-card {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      box-shadow: var(--shadow);
    }}
    .boundary-card {{
      background: var(--surface);
      border: 1px solid #f0d8a8;
      border-radius: 8px;
      padding: 12px;
      box-shadow: var(--shadow);
    }}
    .boundary-card h2 {{
      margin: 0 0 8px;
      font-size: 14px;
      line-height: 1.25;
    }}
    .boundary-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .boundary-list {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}
    .summary-card h2 {{
      margin: 0 0 8px;
      font-size: 14px;
      line-height: 1.25;
    }}
    .metric-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    main {{
      padding: 0 22px 22px;
      display: grid;
      grid-template-columns: minmax(220px, 320px) minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }}
    .sidebar, .detail {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .sidebar {{
      padding: 10px;
      position: sticky;
      top: 12px;
    }}
    .query-button {{
      width: 100%;
      border: 1px solid transparent;
      border-radius: 6px;
      background: transparent;
      color: var(--ink);
      cursor: pointer;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 10px;
      text-align: left;
      min-height: 50px;
    }}
    .query-button:hover {{ background: #f3f6fa; }}
    .query-button.active {{
      border-color: #a8d5d0;
      background: #effbf9;
    }}
    .query-title {{
      font-size: 13px;
      line-height: 1.25;
      font-weight: 680;
      overflow-wrap: anywhere;
    }}
    .pass-pill {{
      font-size: 11px;
      min-width: 42px;
      text-align: center;
      border-radius: 999px;
      padding: 3px 6px;
      color: var(--green);
      background: #ecfdf3;
      border: 1px solid #b7dfc2;
    }}
    .pass-pill.fail {{
      color: var(--red);
      background: #fff1f2;
      border-color: #fecdd3;
    }}
    .detail {{
      padding: 16px;
      min-width: 0;
    }}
    .detail-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .query-text {{
      margin: 6px 0 0;
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
    }}
    .baseline-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .baseline-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }}
    .baseline-card.trace {{
      border-color: #a8d5d0;
      background: #f8fffd;
    }}
    .baseline-name {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 10px;
      font-size: 13px;
      font-weight: 720;
    }}
    .baseline-source {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      margin-bottom: 8px;
      overflow-wrap: anywhere;
    }}
    .section-label {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0;
      font-size: 11px;
      font-weight: 720;
      margin: 12px 0 6px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font-size: 12px;
      line-height: 1.45;
      color: #243244;
    }}
    .trace-list {{
      display: grid;
      gap: 8px;
    }}
    .trace-item {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #fbfcfe;
    }}
    .trace-meta {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 7px;
    }}
    .meta-token {{
      font-size: 11px;
      color: var(--muted);
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 999px;
      padding: 3px 7px;
      max-width: 100%;
      overflow-wrap: anywhere;
    }}
    .meta-token.trace-kind {{
      color: var(--teal);
      border-color: #a8d5d0;
      background: #effbf9;
    }}
    .meta-token.suppressed {{
      color: var(--amber);
      border-color: #f0d8a8;
      background: #fff8ec;
    }}
    .empty {{
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      header {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .status-strip {{
        justify-content: flex-start;
      }}
      .summary, .boundary-grid, main, .baseline-grid {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>AdaMem State Authority Demo</h1>
        <div class="subhead" id="datasetLine"></div>
      </div>
      <div class="status-strip">
        <span class="chip good">API-free</span>
        <span class="chip warn">Not paper evidence</span>
        <span class="chip" id="queryCount"></span>
      </div>
    </header>
    <section class="summary" id="summary"></section>
    <section class="boundary" id="boundary"></section>
    <main>
      <aside class="sidebar" id="queryList"></aside>
      <section class="detail" id="detail"></section>
    </main>
  </div>
  <script type="application/json" id="demo-data">{data}</script>
  <script>
    const payload = JSON.parse(document.getElementById('demo-data').textContent);
    const queries = payload.mode === 'all_queries' ? payload.queries : [payload];
    let activeIndex = 0;

    function text(value) {{
      return value === undefined || value === null || value === '' ? '<none>' : String(value);
    }}

    function escapeHtml(value) {{
      return text(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function passSummary(query) {{
      const passed = query.baselines.filter((baseline) => baseline.passed).length;
      return passed === query.baselines.length ? 'PASS' : 'FAIL';
    }}

    function renderShell() {{
      document.getElementById('datasetLine').textContent = payload.dataset + ' / ' + payload.case_id;
      document.getElementById('queryCount').textContent = queries.length + ' queries';
      renderSummary();
      renderBoundary();
      renderQueryList();
      renderDetail();
    }}

    function renderSummary() {{
      const summary = payload.summary && payload.summary.by_baseline
        ? payload.summary.by_baseline
        : Object.fromEntries((payload.baselines || []).map((baseline) => [
            baseline.name,
            {{
              passed: baseline.passed ? 1 : 0,
              total: 1,
              accuracy: baseline.passed ? 1 : 0,
              state_adjudication_traces: baseline.trace.filter((item) => item.kind === 'state_adjudication').length,
              failed_query_ids: baseline.passed ? [] : [payload.query_id],
            }},
          ]));
      document.getElementById('summary').innerHTML = Object.entries(summary).map(([name, row]) => `
        <article class="summary-card">
          <h2>${{escapeHtml(name)}}</h2>
          <div class="metric-row">
            <span class="chip good">${{row.passed}}/${{row.total}} pass</span>
            <span class="chip">${{Math.round(row.accuracy * 100)}}% support</span>
            <span class="chip">${{row.state_adjudication_traces || 0}} adjudication traces</span>
          </div>
        </article>
      `).join('');
    }}

    function renderBoundary() {{
      const boundary = payload.evidence_boundary || {{}};
      const blocked = boundary.blocked_claims || {{}};
      const blockedItems = Object.entries(blocked).flatMap(([claim, reasons]) =>
        (reasons || []).map((reason) => `${{claim}}: ${{reason}}`)
      );
      document.getElementById('boundary').innerHTML = `
        <article class="boundary-card">
          <h2>Evidence Boundary</h2>
          <div class="boundary-grid">
            <div>
              <div class="section-label">Supported</div>
              ${{renderList(boundary.supported_uses || [])}}
            </div>
            <div>
              <div class="section-label">Blocked</div>
              ${{renderList(blockedItems)}}
            </div>
            <div>
              <div class="section-label">Next Evidence</div>
              ${{renderList(boundary.next_evidence || [])}}
            </div>
          </div>
        </article>
      `;
    }}

    function renderList(items) {{
      if (!items.length) {{
        return '<div class="empty">&lt;none&gt;</div>';
      }}
      return `<ul class="boundary-list">${{items.map((item) => `<li>${{escapeHtml(item)}}</li>`).join('')}}</ul>`;
    }}

    function renderQueryList() {{
      document.getElementById('queryList').innerHTML = queries.map((query, index) => {{
        const status = passSummary(query);
        return `
          <button class="query-button ${{index === activeIndex ? 'active' : ''}}" data-index="${{index}}">
            <span class="query-title">${{escapeHtml(query.query_id)}}</span>
            <span class="pass-pill ${{status === 'PASS' ? '' : 'fail'}}">${{status}}</span>
          </button>
        `;
      }}).join('');
      document.querySelectorAll('.query-button').forEach((button) => {{
        button.addEventListener('click', () => {{
          activeIndex = Number(button.dataset.index);
          renderQueryList();
          renderDetail();
        }});
      }});
    }}

    function renderDetail() {{
      const query = queries[activeIndex];
      document.getElementById('detail').innerHTML = `
        <div class="detail-head">
          <div>
            <h2>${{escapeHtml(query.query_id)}}</h2>
            <p class="query-text">${{escapeHtml(query.query)}}</p>
          </div>
          <span class="chip ${{passSummary(query) === 'PASS' ? 'good' : 'warn'}}">${{passSummary(query)}}</span>
        </div>
        <div class="metric-row">
          <span class="chip">expected: ${{escapeHtml((query.expected_substrings || []).join(', ') || '<none>')}}</span>
          <span class="chip">forbidden: ${{escapeHtml((query.forbidden_substrings || []).join(', ') || '<none>')}}</span>
          <span class="chip">top_k: ${{escapeHtml(query.top_k)}}</span>
        </div>
        <div class="baseline-grid">
          ${{query.baselines.map(renderBaseline).join('')}}
        </div>
      `;
    }}

    function renderBaseline(baseline) {{
      const hasTraceNotice = baseline.trace.some((item) => item.kind === 'state_adjudication');
      return `
        <article class="baseline-card ${{hasTraceNotice ? 'trace' : ''}}">
          <div class="baseline-name">
            <span>${{escapeHtml(baseline.name)}}</span>
            <span class="pass-pill ${{baseline.passed ? '' : 'fail'}}">${{baseline.passed ? 'PASS' : 'FAIL'}}</span>
          </div>
          <div class="baseline-source">
            ${{escapeHtml(baseline.source_name || baseline.category)}} · ${{escapeHtml(baseline.implementation_status || 'unknown')}}
          </div>
          <div class="section-label">Retrieved</div>
          ${{baseline.retrieved.length ? baseline.retrieved.map((item) => `<pre>${{escapeHtml(item)}}</pre>`).join('') : '<div class="empty">&lt;none&gt;</div>'}}
          <div class="section-label">Trace</div>
          <div class="trace-list">
            ${{baseline.trace.length ? baseline.trace.map(renderTrace).join('') : '<div class="empty">&lt;none&gt;</div>'}}
          </div>
        </article>
      `;
    }}

    function renderTrace(item) {{
      const metadata = item.metadata || {{}};
      const source = metadata.source_observation_label || '<none>';
      const suppressed = metadata.adjudicated_source_observation_label || '<none>';
      return `
        <div class="trace-item">
          <div class="trace-meta">
            <span class="meta-token trace-kind">${{escapeHtml(item.kind)}}</span>
            <span class="meta-token">slot=${{escapeHtml(metadata.state_slot || '<none>')}}</span>
            <span class="meta-token">source=${{escapeHtml(source)}}</span>
            <span class="meta-token suppressed">suppressed=${{escapeHtml(suppressed)}}</span>
            <span class="meta-token">score=${{escapeHtml(item.score)}}</span>
          </div>
          <pre>${{escapeHtml(item.content)}}</pre>
        </div>
      `;
    }}

    renderShell();
  </script>
</body>
</html>
"""


def _json_for_script(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return text.replace("</", "<\\/")
