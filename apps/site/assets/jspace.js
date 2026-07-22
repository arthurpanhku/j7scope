/* Shared client logic for the J7Scope static platform.
 *
 * One implementation of read-out / rigor rendering, trace loading, and
 * paper-grade export (SVG / PNG / JSON / BibTeX), reused by gallery, replay,
 * compare, and live pages. Trace files are fetched via *relative* paths
 * ("traces/<id>/...") so the same pages work served by the sidecar (at /) and
 * on GitHub Pages (under a project subpath).
 */
window.JSpace = (function () {
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function pct(x) { return (clamp(x, 0, 1) * 100).toFixed(1) + "%"; }
  function maxScore(list) { var m = -1e9; list.forEach(function (x) { if (x.score > m) m = x.score; }); return m; }

  // ---- trace loading (relative paths) ----
  function fetchIndex() {
    return fetch("traces/index.json").then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; });
  }
  function fetchTrace(id) {
    return Promise.all([
      fetch("traces/" + id + "/manifest.json").then(function (r) { return r.json(); }),
      fetch("traces/" + id + "/tokens.jsonl").then(function (r) { return r.text(); }),
      fetch("traces/" + id + "/metrics.json").then(function (r) { return r.json(); }),
      fetch("traces/" + id + "/align.json").then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; })
    ]).then(function (res) {
      return {
        manifest: res[0],
        tokens: res[1].trim().split("\n").filter(Boolean).map(JSON.parse),
        metrics: res[2],
        align: res[3]
      };
    });
  }

  // ---- rendering ----
  function renderStream(el, tokens, curIdx, onClick) {
    el.innerHTML = tokens.map(function (t, i) {
      return '<span class="tok' + (i === curIdx ? " cur" : "") + '" data-i="' + i + '">' + esc(t) + " </span>";
    }).join("") + '<span class="caret">▍</span>';
    if (onClick) {
      Array.prototype.forEach.call(el.querySelectorAll(".tok"), function (s) {
        s.style.cursor = "pointer";
        s.onclick = function () { onClick(parseInt(s.getAttribute("data-i"), 10)); };
      });
    }
    el.scrollTop = el.scrollHeight;
  }

  function renderColumn(el, list, matchSet) {
    if (!list || !list.length) { el.innerHTML = '<div class="empty">—</div>'; return; }
    var hi = maxScore(list), lo = hi - 6;
    el.innerHTML = list.map(function (item) {
      var frac = clamp((item.score - lo) / ((hi - lo) || 1), 0.04, 1);
      var match = matchSet && matchSet[item.token] ? " match" : "";
      return '<div class="row' + match + '"><div class="word">' + esc(item.token) + "</div>" +
        '<div class="bar"><span style="width:' + pct(frac) + '"></span></div>' +
        '<div class="score">' + item.score.toFixed(1) + "</div></div>";
    }).join("");
  }

  // Build the rigor strip HTML (self-contained) into a container element.
  function renderRigor(container, tok) {
    var r = tok && tok.rigor;
    if (!r) { container.style.display = "none"; return; }
    container.style.display = "";
    var obs = r.cross_lang_overlap, nb = r["null"], same = r.same_lang_baseline;
    var above = obs > nb.p95;
    var ci = r.sharedness.ci95;
    container.className = "rigor";
    container.innerHTML =
      '<div class="top"><span class="tk">' + esc(tok.token) + (tok.concept ? "  · " + esc(tok.concept) : "") + '</span>' +
      '<span class="verdict ' + (above ? "above" : "within") + '">' + (above ? "above null · 真信号" : "within null · 噪声内") + '</span>' +
      '<span class="caption">跨语言概念重叠 vs 打乱配对 null 基线（越过 null 带才算真信号）</span></div>' +
      '<div class="meter' + (above ? " aboveNull" : "") + '">' +
        '<div class="nullband" style="left:' + pct(nb.p05) + ';width:' + pct(Math.max(0, nb.p95 - nb.p05)) + '"></div>' +
        '<div class="nullmean" style="left:' + pct(nb.mean) + '"></div>' +
        '<div class="obs" style="width:' + pct(obs) + '"></div>' +
        '<div class="obsmark" style="left:' + pct(obs) + '"></div></div>' +
      '<div class="scale"><span>0</span><span>concept overlap</span><span>1</span></div>' +
      '<div class="nums">' +
        '<div><span>observed</span> <b>' + obs.toFixed(2) + '</b></div>' +
        '<div><span>null p05–p95</span> <b>' + nb.p05.toFixed(2) + "–" + nb.p95.toFixed(2) + '</b></div>' +
        '<div><span>same-lang ceiling</span> <b>' + same.toFixed(2) + '</b></div>' +
        '<div><span>sharedness</span> <b>' + r.sharedness.value.toFixed(2) + '</b> ' +
          (ci ? '<span>CI ' + ci[0].toFixed(2) + "–" + ci[1].toFixed(2) + '</span>' : "") + '</div>' +
      '</div>';
  }

  // ---- export ----
  function download(name, mime, text) {
    var blob = new Blob([text], { type: mime });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url; a.download = name; a.click();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function exportTokenJSON(trace, idx) {
    var tok = trace.tokens[idx];
    var payload = {
      trace_id: trace.manifest.trace_id, model: trace.manifest.model,
      layer: trace.manifest.layer, language: trace.manifest.language,
      seq: tok.seq, token: tok.token, readout: tok.readout, rigor: tok.rigor
    };
    download(trace.manifest.trace_id + "-token" + idx + ".json",
      "application/json", JSON.stringify(payload, null, 2));
  }

  function bibtex(trace, deepUrl) {
    var m = trace.manifest;
    var year = (m.capture && m.capture.created_at || "").slice(0, 4) || new Date().getFullYear();
    var key = "j7scope_" + String(m.trace_id).replace(/[^a-zA-Z0-9]/g, "");
    var doi = m.doi ? "  doi          = {" + m.doi + "},\n" : "  note         = {No DOI yet; demo/preview trace},\n";
    return "@software{" + key + ",\n" +
      "  title        = {J7Scope J-Space trace: " + m.trace_id + "},\n" +
      "  author       = {J7Scope contributors},\n" +
      "  year         = {" + year + "},\n" +
      "  howpublished = {\\url{" + deepUrl + "}},\n" +
      "  note         = {model=" + m.model + ", layer=" + m.layer + ", language=" + m.language + "},\n" +
      doi + "}\n";
  }
  function exportBibTeX(trace, deepUrl) {
    download(trace.manifest.trace_id + ".bib", "text/plain", bibtex(trace, deepUrl));
  }

  // Paper-grade SVG figure of the rigor strip + top-3 concepts per language.
  function tokenSVG(trace, idx) {
    var m = trace.manifest, tok = trace.tokens[idx], r = tok.rigor;
    var W = 720, H = 300, pad = 24, barY = 70, barW = W - 2 * pad, barH = 22;
    var above = r.cross_lang_overlap > r["null"].p95;
    var col = above ? "#57c98a" : "#b98cff";
    function X(v) { return pad + clamp(v, 0, 1) * barW; }
    function rows(list, x, color) {
      return list.slice(0, 3).map(function (it, i) {
        return '<text x="' + x + '" y="' + (200 + i * 24) + '" fill="' + color + '" font-size="15">' +
          esc(it.token) + '</text>';
      }).join("");
    }
    var s = [];
    s.push('<svg xmlns="http://www.w3.org/2000/svg" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" font-family="-apple-system, Segoe UI, PingFang SC, sans-serif">');
    s.push('<rect width="' + W + '" height="' + H + '" fill="#0e0f13"/>');
    s.push('<text x="' + pad + '" y="30" fill="#e7e9f0" font-size="17" font-weight="700">' + esc(tok.token) + (tok.concept ? "  ·  " + esc(tok.concept) : "") + '</text>');
    s.push('<text x="' + pad + '" y="50" fill="#8b90a3" font-size="12">' + esc(m.trace_id) + " · " + esc(m.model) + " · layer " + m.layer + " · token " + idx + '</text>');
    // meter
    s.push('<rect x="' + pad + '" y="' + barY + '" width="' + barW + '" height="' + barH + '" rx="6" fill="#1c1f2b"/>');
    s.push('<rect x="' + X(r["null"].p05) + '" y="' + barY + '" width="' + (X(r["null"].p95) - X(r["null"].p05)) + '" height="' + barH + '" fill="#3a4055"/>');
    s.push('<rect x="' + pad + '" y="' + barY + '" width="' + (X(r.cross_lang_overlap) - pad) + '" height="' + barH + '" rx="6" fill="' + col + '" opacity="0.85"/>');
    s.push('<line x1="' + X(r["null"].mean) + '" y1="' + (barY - 4) + '" x2="' + X(r["null"].mean) + '" y2="' + (barY + barH + 4) + '" stroke="#6b7288" stroke-width="2"/>');
    s.push('<text x="' + pad + '" y="' + (barY + barH + 18) + '" fill="#8b90a3" font-size="11">0</text>');
    s.push('<text x="' + (W - pad) + '" y="' + (barY + barH + 18) + '" fill="#8b90a3" font-size="11" text-anchor="end">1  concept overlap</text>');
    // numbers
    var line = "observed " + r.cross_lang_overlap.toFixed(2) + "   ·   null " + r["null"].p05.toFixed(2) + "–" + r["null"].p95.toFixed(2) +
      "   ·   sharedness " + r.sharedness.value.toFixed(2) + (r.sharedness.ci95 ? " (CI " + r.sharedness.ci95[0].toFixed(2) + "–" + r.sharedness.ci95[1].toFixed(2) + ")" : "");
    s.push('<text x="' + pad + '" y="' + (barY + barH + 42) + '" fill="#e7e9f0" font-size="13">' + esc(line) + '</text>');
    s.push('<text x="' + pad + '" y="180" fill="#f2a65a" font-size="12">中文</text>');
    s.push('<text x="' + (W / 2) + '" y="180" fill="#6aa9ff" font-size="12">English</text>');
    s.push(rows(tok.readout.zh, pad, "#f2a65a"));
    s.push(rows(tok.readout.en, W / 2, "#6aa9ff"));
    s.push('<text x="' + (W - pad) + '" y="' + (H - 12) + '" fill="#8b90a3" font-size="10" text-anchor="end">J7Scope · github.com/arthurpanhku/j7scope</text>');
    s.push('</svg>');
    return s.join("");
  }
  function exportSVG(trace, idx) {
    download(trace.manifest.trace_id + "-token" + idx + ".svg", "image/svg+xml", tokenSVG(trace, idx));
  }
  function exportPNG(trace, idx) {
    var svg = tokenSVG(trace, idx);
    var img = new Image();
    var url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    img.onload = function () {
      var scale = 2, canvas = document.createElement("canvas");
      canvas.width = img.width * scale; canvas.height = img.height * scale;
      var ctx = canvas.getContext("2d"); ctx.scale(scale, scale); ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      canvas.toBlob(function (blob) {
        var u = URL.createObjectURL(blob), a = document.createElement("a");
        a.href = u; a.download = trace.manifest.trace_id + "-token" + idx + ".png"; a.click();
        setTimeout(function () { URL.revokeObjectURL(u); }, 1000);
      });
    };
    img.src = url;
  }

  return {
    esc: esc, clamp: clamp, pct: pct,
    fetchIndex: fetchIndex, fetchTrace: fetchTrace,
    renderStream: renderStream, renderColumn: renderColumn, renderRigor: renderRigor,
    exportTokenJSON: exportTokenJSON, exportBibTeX: exportBibTeX,
    exportSVG: exportSVG, exportPNG: exportPNG, tokenSVG: tokenSVG
  };
})();
