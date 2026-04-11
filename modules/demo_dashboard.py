import json
import math
import random
from datetime import date, timedelta


def _scaled_daily_values(target_total: int, days: int, kind: str) -> list[int]:
    rng = random.Random(17 if kind == "users" else 29)
    end = date(2026, 4, 10)
    start = end - timedelta(days=days - 1)
    dates = [start + timedelta(days=i) for i in range(days)]

    values = []
    for i, day in enumerate(dates):
        if kind == "users":
            base = 7.2 + 0.018 * i + 2.0 / (1 + math.exp(-(i - 95) / 23))
            weekly = [0.92, 1.0, 1.04, 1.06, 1.02, 0.90, 0.84][day.weekday()]
            noise = rng.gauss(0, 0.5)
            values.append(max(3.0, base * weekly + noise))
        else:
            base = 105 + 0.28 * i + 34 / (1 + math.exp(-(i - 105) / 26))
            weekly = [0.93, 1.0, 1.03, 1.05, 1.02, 0.91, 0.85][day.weekday()]
            noise = rng.gauss(0, 9)
            values.append(max(35.0, base * weekly + noise))

    scale = target_total / sum(values)
    scaled = [int(round(value * scale)) for value in values]

    while sum(scaled) != target_total:
        idx = (abs(sum(scaled) - target_total) * 7 + 13) % len(scaled)
        if sum(scaled) < target_total:
            scaled[idx] += 1
        elif scaled[idx] > 1:
            scaled[idx] -= 1

    return scaled


def _month_ticks(days: int) -> list[dict[str, str | int]]:
    end = date(2026, 4, 10)
    start = end - timedelta(days=days - 1)
    dates = [start + timedelta(days=i) for i in range(days)]
    ticks = []
    for idx, day in enumerate(dates):
        if day.day == 1:
            ticks.append({"index": idx, "label": day.strftime("%b 1")})
    return ticks


def build_growth_dashboard_page() -> str:
    days = 180
    users = _scaled_daily_values(2200, days, "users")
    queries = _scaled_daily_values(30000, days, "queries")
    month_ticks = _month_ticks(days)

    payload = {
        "users": users,
        "queries": queries,
        "monthTicks": month_ticks,
        "totals": {
            "users": sum(users),
            "queries": sum(queries),
            "days": days,
        },
        "ranges": {
            "usersMax": 20,
            "usersStep": 5,
            "queriesMax": 300,
            "queriesStep": 75,
        },
    }

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Growth Dashboard</title>
    <style>
      :root {{
        --bg: #f8fafc;
        --card: #ffffff;
        --border: #e5e7eb;
        --grid: #edf2f7;
        --text: #111827;
        --muted: #6b7280;
        --blue: #3b82f6;
        --blue-dark: #2563eb;
        --blue-soft: #dbeafe;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: var(--bg);
        color: var(--text);
      }}

      .page {{
        max-width: 1440px;
        margin: 0 auto;
        padding: 28px 24px 40px;
      }}

      .eyebrow {{
        color: var(--muted);
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.02em;
      }}

      h1 {{
        margin: 8px 0 10px;
        font-size: clamp(34px, 4vw, 52px);
        line-height: 1.04;
      }}

      .subtitle {{
        color: var(--muted);
        font-size: 18px;
        margin-bottom: 22px;
      }}

      .toolbar {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 22px;
      }}

      .button {{
        appearance: none;
        border: 1px solid var(--border);
        background: var(--card);
        color: var(--text);
        border-radius: 12px;
        padding: 11px 16px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: 160ms ease;
      }}

      .button:hover {{
        border-color: #bfdbfe;
        background: #eff6ff;
      }}

      .button.primary {{
        background: var(--blue);
        border-color: var(--blue);
        color: white;
      }}

      .button.primary:hover {{
        background: var(--blue-dark);
        border-color: var(--blue-dark);
      }}

      #status-pill {{
        display: inline-flex;
        align-items: center;
        min-height: 42px;
        padding: 0 14px;
        border-radius: 999px;
        background: #eff6ff;
        color: var(--blue-dark);
        font-size: 13px;
        font-weight: 600;
      }}

      .dashboard-shell {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 28px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
      }}

      .cards {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        border-bottom: 1px solid var(--border);
      }}

      .card {{
        padding: 24px 28px;
        min-height: 118px;
        border-right: 1px solid var(--border);
      }}

      .card:last-child {{
        border-right: none;
      }}

      .card-label {{
        color: var(--muted);
        font-size: 15px;
        margin-bottom: 10px;
      }}

      .card-value {{
        font-size: clamp(34px, 4vw, 56px);
        font-weight: 700;
        line-height: 1;
      }}

      .card-tag {{
        margin-top: 10px;
        color: var(--blue);
        font-size: 13px;
        font-weight: 600;
      }}

      .chart-card {{
        padding: 28px 24px 18px;
        border-bottom: 1px solid var(--border);
      }}

      .chart-card:last-child {{
        border-bottom: none;
      }}

      .chart-header {{
        display: flex;
        justify-content: space-between;
        gap: 20px;
        align-items: flex-start;
        margin-bottom: 18px;
      }}

      .chart-title {{
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 6px;
      }}

      .chart-subtitle, .chart-meta {{
        color: var(--muted);
        font-size: 14px;
      }}

      .chart-meta {{
        text-align: right;
      }}

      .chart-svg {{
        width: 100%;
        height: auto;
        display: block;
      }}

      .footer-note {{
        padding: 20px 24px 24px;
        color: var(--muted);
        font-size: 13px;
      }}

      @media (max-width: 980px) {{
        .cards {{
          grid-template-columns: 1fr;
        }}

        .card {{
          border-right: none;
          border-bottom: 1px solid var(--border);
        }}

        .card:last-child {{
          border-bottom: none;
        }}

        .chart-header {{
          flex-direction: column;
          gap: 8px;
        }}

        .chart-meta {{
          text-align: left;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="page">
      <div class="eyebrow">Growth Dashboard</div>
      <h1>Users and Queries Over Time</h1>
      <div class="subtitle">
        Shadcn-style sample dashboard with blue daily bar charts across the last 6 months.
      </div>

      <div class="toolbar">
        <button class="button primary" id="download-png">Download PNG</button>
        <button class="button" id="download-svg">Download SVG</button>
        <button class="button" id="copy-image">Copy Image</button>
        <button class="button" id="download-json">Download Data</button>
        <span id="status-pill">Ready</span>
      </div>

      <div class="dashboard-shell" id="dashboard-shell">
        <div class="cards">
          <div class="card">
            <div class="card-label">Total Users</div>
            <div class="card-value">2,200</div>
            <div class="card-tag">6-month total</div>
          </div>
          <div class="card">
            <div class="card-label">Total Queries</div>
            <div class="card-value">30,000</div>
            <div class="card-tag">6-month total</div>
          </div>
          <div class="card">
            <div class="card-label">Time Window</div>
            <div class="card-value">180 days</div>
            <div class="card-tag">daily activity view</div>
          </div>
        </div>

        <div class="chart-card">
          <div class="chart-header">
            <div>
              <div class="chart-title">Daily Users</div>
              <div class="chart-subtitle">New users per day over the last 6 months</div>
            </div>
            <div class="chart-meta" id="users-meta"></div>
          </div>
          <svg class="chart-svg" id="users-chart" viewBox="0 0 1320 330" role="img" aria-label="Daily users bar chart"></svg>
        </div>

        <div class="chart-card">
          <div class="chart-header">
            <div>
              <div class="chart-title">Daily Queries</div>
              <div class="chart-subtitle">Queries processed per day over the last 6 months</div>
            </div>
            <div class="chart-meta" id="queries-meta"></div>
          </div>
          <svg class="chart-svg" id="queries-chart" viewBox="0 0 1320 330" role="img" aria-label="Daily queries bar chart"></svg>
        </div>

        <div class="footer-note">
          Sample startup pitch chart. Daily values are synthetic, but totals and pacing are normalized to a realistic 6-month growth pattern.
        </div>
      </div>
    </div>

    <script>
      const DATA = {json.dumps(payload)};

      function formatNumber(value) {{
        return new Intl.NumberFormat('en-US').format(value);
      }}

      function setStatus(text) {{
        document.getElementById('status-pill').textContent = text;
      }}

      function buildChart(svgId, values, config, color) {{
        const svg = document.getElementById(svgId);
        const width = 1320;
        const height = 330;
        const margins = {{ top: 12, right: 24, bottom: 44, left: 46 }};
        const plotWidth = width - margins.left - margins.right;
        const plotHeight = height - margins.top - margins.bottom;
        const baseY = margins.top + plotHeight;
        const gap = plotWidth / values.length;
        const barWidth = Math.max(3, gap - 2);

        const parts = [];
        parts.push(`<rect x="0" y="0" width="${{width}}" height="${{height}}" fill="white"></rect>`);

        for (let tick = config.step; tick <= config.max; tick += config.step) {{
          const y = margins.top + plotHeight - (tick / config.max) * plotHeight;
          parts.push(`<line x1="${{margins.left}}" y1="${{y}}" x2="${{width - margins.right}}" y2="${{y}}" stroke="#edf2f7" stroke-width="1"></line>`);
          parts.push(`<text x="${{margins.left - 10}}" y="${{y + 4}}" text-anchor="end" fill="#6b7280" font-size="12" font-family="Inter, Arial, sans-serif">${{tick}}</text>`);
        }}

        for (const tick of DATA.monthTicks) {{
          const x = margins.left + tick.index * gap + gap / 2;
          parts.push(`<line x1="${{x}}" y1="${{margins.top}}" x2="${{x}}" y2="${{baseY}}" stroke="#f3f4f6" stroke-width="1"></line>`);
          parts.push(`<text x="${{x}}" y="${{baseY + 24}}" text-anchor="middle" fill="#6b7280" font-size="12" font-family="Inter, Arial, sans-serif">${{tick.label}}</text>`);
        }}

        values.forEach((value, index) => {{
          const x = margins.left + index * gap + (gap - barWidth) / 2;
          const heightValue = (value / config.max) * plotHeight;
          const y = baseY - heightValue;
          const opacity = index % 7 >= 5 ? 0.72 : 0.86;
          parts.push(`<rect x="${{x.toFixed(2)}}" y="${{y.toFixed(2)}}" width="${{barWidth.toFixed(2)}}" height="${{heightValue.toFixed(2)}}" rx="2.4" fill="${{color}}" opacity="${{opacity}}"></rect>`);
        }});

        parts.push(`<text x="${{margins.left}}" y="${{height - 6}}" fill="#6b7280" font-size="12" font-family="Inter, Arial, sans-serif">Oct 2025</text>`);
        parts.push(`<text x="${{width - margins.right}}" y="${{height - 6}}" text-anchor="end" fill="#6b7280" font-size="12" font-family="Inter, Arial, sans-serif">Apr 2026</text>`);
        svg.innerHTML = parts.join('');
      }}

      function buildFullSvgMarkup() {{
        const usersSvg = document.getElementById('users-chart').innerHTML;
        const queriesSvg = document.getElementById('queries-chart').innerHTML;

        return `
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1220" viewBox="0 0 1600 1220">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="36" y="42" fill="#6b7280" font-size="16" font-family="Inter, Arial, sans-serif">Growth Dashboard</text>
  <text x="36" y="88" fill="#111827" font-size="42" font-weight="700" font-family="Inter, Arial, sans-serif">Users and Queries Over Time</text>
  <text x="36" y="118" fill="#6b7280" font-size="18" font-family="Inter, Arial, sans-serif">Daily bars across the last 6 months, scaled to 2.2k total users and 30k total queries.</text>

  <rect x="36" y="150" width="490" height="110" rx="22" fill="#ffffff" stroke="#e5e7eb"/>
  <rect x="555" y="150" width="490" height="110" rx="22" fill="#ffffff" stroke="#e5e7eb"/>
  <rect x="1074" y="150" width="490" height="110" rx="22" fill="#ffffff" stroke="#e5e7eb"/>

  <text x="64" y="188" fill="#6b7280" font-size="16" font-family="Inter, Arial, sans-serif">Total Users</text>
  <text x="64" y="236" fill="#111827" font-size="44" font-weight="700" font-family="Inter, Arial, sans-serif">2,200</text>
  <text x="474" y="188" text-anchor="end" fill="#2563eb" font-size="13" font-family="Inter, Arial, sans-serif">6-month total</text>

  <text x="583" y="188" fill="#6b7280" font-size="16" font-family="Inter, Arial, sans-serif">Total Queries</text>
  <text x="583" y="236" fill="#111827" font-size="44" font-weight="700" font-family="Inter, Arial, sans-serif">30,000</text>
  <text x="993" y="188" text-anchor="end" fill="#2563eb" font-size="13" font-family="Inter, Arial, sans-serif">6-month total</text>

  <text x="1102" y="188" fill="#6b7280" font-size="16" font-family="Inter, Arial, sans-serif">Time Window</text>
  <text x="1102" y="236" fill="#111827" font-size="44" font-weight="700" font-family="Inter, Arial, sans-serif">180 days</text>
  <text x="1512" y="188" text-anchor="end" fill="#2563eb" font-size="13" font-family="Inter, Arial, sans-serif">daily activity view</text>

  <rect x="36" y="292" width="1528" height="404" rx="26" fill="#ffffff" stroke="#e5e7eb"/>
  <text x="68" y="332" fill="#111827" font-size="22" font-weight="700" font-family="Inter, Arial, sans-serif">Daily Users</text>
  <text x="68" y="358" fill="#6b7280" font-size="14" font-family="Inter, Arial, sans-serif">New users per day over the last 6 months</text>
  <text x="1498" y="332" text-anchor="end" fill="#6b7280" font-size="14" font-family="Inter, Arial, sans-serif">${{document.getElementById('users-meta').innerText.replace(/\\n/g, ' · ')}}</text>
  <g transform="translate(68, 380)">${{usersSvg}}</g>

  <rect x="36" y="724" width="1528" height="404" rx="26" fill="#ffffff" stroke="#e5e7eb"/>
  <text x="68" y="764" fill="#111827" font-size="22" font-weight="700" font-family="Inter, Arial, sans-serif">Daily Queries</text>
  <text x="68" y="790" fill="#6b7280" font-size="14" font-family="Inter, Arial, sans-serif">Queries processed per day over the last 6 months</text>
  <text x="1498" y="764" text-anchor="end" fill="#6b7280" font-size="14" font-family="Inter, Arial, sans-serif">${{document.getElementById('queries-meta').innerText.replace(/\\n/g, ' · ')}}</text>
  <g transform="translate(68, 812)">${{queriesSvg}}</g>

  <text x="36" y="1182" fill="#6b7280" font-size="13" font-family="Inter, Arial, sans-serif">Sample startup pitch chart. Daily values are synthetic, but totals and pacing are normalized to a realistic 6-month growth pattern.</text>
</svg>`;
      }}

      async function svgToPngBlob() {{
        const svgMarkup = buildFullSvgMarkup();
        const blob = new Blob([svgMarkup], {{ type: 'image/svg+xml;charset=utf-8' }});
        const url = URL.createObjectURL(blob);

        try {{
          const img = new Image();
          img.decoding = 'async';
          const loaded = new Promise((resolve, reject) => {{
            img.onload = resolve;
            img.onerror = reject;
          }});
          img.src = url;
          await loaded;

          const canvas = document.createElement('canvas');
          canvas.width = 1600;
          canvas.height = 1220;
          const ctx = canvas.getContext('2d');
          ctx.fillStyle = '#f8fafc';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0);

          return await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
        }} finally {{
          URL.revokeObjectURL(url);
        }}
      }}

      function download(filename, blob) {{
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = filename;
        anchor.click();
        URL.revokeObjectURL(url);
      }}

      function render() {{
        buildChart('users-chart', DATA.users, {{ max: DATA.ranges.usersMax, step: DATA.ranges.usersStep }}, '#3b82f6');
        buildChart('queries-chart', DATA.queries, {{ max: DATA.ranges.queriesMax, step: DATA.ranges.queriesStep }}, '#2563eb');

        const usersAvg = (DATA.totals.users / DATA.totals.days).toFixed(1);
        const usersPeak = Math.max(...DATA.users);
        const queriesAvg = (DATA.totals.queries / DATA.totals.days).toFixed(1);
        const queriesPeak = Math.max(...DATA.queries);

        document.getElementById('users-meta').innerText = `avg/day ${{usersAvg}}\\npeak/day ${{usersPeak}}`;
        document.getElementById('queries-meta').innerText = `avg/day ${{queriesAvg}}\\npeak/day ${{queriesPeak}}`;
      }}

      document.getElementById('download-svg').addEventListener('click', () => {{
        const svgMarkup = buildFullSvgMarkup();
        download('growth-dashboard.svg', new Blob([svgMarkup], {{ type: 'image/svg+xml;charset=utf-8' }}));
        setStatus('SVG downloaded');
      }});

      document.getElementById('download-png').addEventListener('click', async () => {{
        const blob = await svgToPngBlob();
        download('growth-dashboard.png', blob);
        setStatus('PNG downloaded');
      }});

      document.getElementById('download-json').addEventListener('click', () => {{
        download('growth-dashboard-data.json', new Blob([JSON.stringify(DATA, null, 2)], {{ type: 'application/json' }}));
        setStatus('Data downloaded');
      }});

      document.getElementById('copy-image').addEventListener('click', async () => {{
        try {{
          const blob = await svgToPngBlob();
          if (!navigator.clipboard || !window.ClipboardItem) {{
            download('growth-dashboard.png', blob);
            setStatus('Clipboard unavailable, PNG downloaded instead');
            return;
          }}
          await navigator.clipboard.write([new ClipboardItem({{ 'image/png': blob }})]);
          setStatus('Image copied to clipboard');
        }} catch (error) {{
          console.error(error);
          setStatus('Could not copy image');
        }}
      }});

      render();
    </script>
  </body>
</html>
"""
