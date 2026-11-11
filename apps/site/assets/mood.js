(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var Signal = window.MoodSignal;
  var I18n = window.MoodI18n, t = I18n.t, message = I18n.message, raw = I18n.raw;
  var lastResult, lastOptions, lastUpdate;
  var mode = "text", events = null, timer = null, history = [], text = "", readout = null;
  var activeRequest = null, lastSeq = -1, connected = false, demoPaused = false, demoIndex = 0;
  var requestController = null, lastTokenAt = 0, stale = false, generationActive = false;
  var modeVersion = 0;
  var seenToolEvents = new Set();
  var demos = [
    "今天的事情很顺利，我很开心，也很期待明天！",
    "I feel relaxed and grateful. It has been a happy day.",
    "心里有些开心，却也有一点难过。",
    "今天有点难过，觉得很疲惫，也有些失望。",
    "I'm not happy. I feel frustrated and tired.",
    "现在不那么焦虑了，慢慢放松，也开始期待新的事情。",
  ];
  var demosEn = [
    "Things went well today. I feel happy and hopeful about tomorrow!",
    "I feel relaxed and grateful. It has been a happy day.",
    "I feel happy, but also a little sad.",
    "I feel sad, tired and a little disappointed today.",
    "I'm not happy. I feel frustrated and tired.",
    "I am not as anxious now. I feel relaxed and hopeful.",
  ];
  var labels = {
    good: ["好心情", "此刻，有一点晴朗", "观察到了偏积极的情绪信号。", "M 52 94 Q 80 128 108 94"],
    bad: ["坏心情", "此刻，有一点阴天", "观察到了偏消极的情绪信号。", "M 52 111 Q 80 80 108 111"],
    mixed: ["心情有点复杂", "晴与阴，都在这里", "正负信号接近，暂不判断好坏。", "M 55 103 Q 80 96 105 103"],
    unknown: ["等待信号", "等一句话，感受一下", "没有足够的情绪关键词，暂不判断。", "M 57 100 L 103 100"],
  };

  function connection(label, status) {
    message("connection-text", label);
    $("connection").dataset.status = status || "ready";
  }

  function clock(timestamp) {
    return new Date(timestamp).toLocaleTimeString(I18n.language() === "en" ? "en-GB" : "zh-CN", { hour12: false });
  }

  function drawHistory() {
    $("history-bars").replaceChildren();
    history.forEach(function (point) {
      var slot = document.createElement("span"), bar = document.createElement("span");
      slot.className = "history-slot";
      bar.className = "history-bar" + (point.score === null ? " unknown" : point.score < 0 ? " negative" : "");
      bar.style.height = (point.score === null ? 2 : Math.max(3, Math.abs(point.score) * 36)) + "px";
      slot.title = clock(point.time) + " · " + t(labels[point.state][0]) +
        (point.score === null ? "" : " · " + t("情绪指数 {score}", { score: Math.round(point.score * 100) }));
      slot.appendChild(bar);
      $("history-bars").appendChild(slot);
    });
    $("history-empty").hidden = history.length > 0;
    $("history-chart").setAttribute("aria-label", history.length ? t("最近 {count} 次变化：{states}", {
      count: history.length,
      states: history.map(function (point) { return t(labels[point.state][0]); }).join(I18n.language() === "en" ? ", " : "、"),
    }) : t("暂无情绪变化记录"));
  }

  function render(result, options) {
    options = options || {};
    lastResult = result;
    lastOptions = options;
    var copy = labels[result.state];
    $("mood-panel").dataset.mood = result.state;
    message("state-label", options.label || copy[0]);
    message("state-kicker", copy[1]);
    message("state-detail", options.detail || copy[2]);
    $("face-mouth").setAttribute("d", copy[3]);
    message("index-label", "情绪指数 {score}", { score: result.score === null ? "—" : (result.score > 0 ? "+" : "") + Math.round(result.score * 100) });
    $("valence-marker").hidden = result.score === null;
    if (result.score !== null) $("valence-marker").style.left = (50 + 48 * result.score) + "%";
    $("evidence").replaceChildren();
    var unique = new Map();
    result.evidence.forEach(function (entry) { unique.set(entry.text + entry.polarity, entry); });
    message("evidence-count", "{count} 个关键词", { count: result.evidence.length });
    if (!unique.size) {
      var hint = document.createElement("span");
      hint.className = "empty-hint";
      hint.textContent = t("还没有可判断的情绪词。");
      $("evidence").appendChild(hint);
    }
    Array.from(unique.values()).slice(0, 12).forEach(function (entry) {
      var chip = document.createElement("span");
      chip.className = "evidence-chip" + (entry.polarity < 0 ? " negative" : "");
      chip.textContent = (entry.polarity > 0 ? "+ " : "− ") + entry.text;
      $("evidence").appendChild(chip);
    });
    if (options.record) {
      lastUpdate = options.timestamp || Date.now();
      message("update-time", "{time} 更新", { time: clock(lastUpdate) });
      history.push({ state: result.state, score: result.score, time: lastUpdate });
      history = history.slice(-32);
      drawHistory();
    }
  }

  function unknown(detail, label) {
    render(Signal.classifyText(""), { detail: detail, label: label });
  }

  function showVectorWords(words, source, status) {
    var list = $("vector-words");
    list.replaceChildren();
    message("vector-source", source);
    message("vector-status", status);
    for (var entry of words.slice(0, 6)) {
      if (!entry || typeof entry.word !== "string" || !Number.isFinite(entry.score)) continue;
      var chip = document.createElement("li");
      chip.append(entry.word);
      var score = document.createElement("small");
      score.textContent = entry.score.toFixed(2);
      chip.append(score);
      list.append(chip);
    }
  }

  function analyze(record, timestamp) {
    var isReadout = mode === "live" && $("live-basis").value === "readout";
    var observed = mode === "live" ? text.slice(-500) : text;
    if (observed) raw("observed-text", observed);
    else message("observed-text", "暂时没有文字。");
    message("observed-heading", isReadout ? "同期输出文字 · 不参与当前判断" : "正在观察的文字");
    message("signal-note", isReadout ? "按 J-lens top-k 情绪词排名加权；探索性信号，不代表模型真实情绪。" : "根据文字中的情绪关键词判断，不代表真实心理状态。");
    render(isReadout ? Signal.classifyReadout(readout) : Signal.classifyText(observed), { record: record, timestamp: timestamp });
  }

  function demoStep() {
    var samples = I18n.language() === "en" ? demosEn : demos;
    text = samples[demoIndex++ % samples.length];
    analyze(true);
  }

  function sendEnabled() {
    $("send-prompt").disabled = !connected || !!requestController || generationActive;
  }

  function openLive() {
    var epoch = modeVersion;
    var toolsMode = mode === "tools";
    var demoReadout = false;
    connection("正在连接…", "waiting");
    message("live-status", "等待当前服务的实时事件流…");
    message("model-label", "等待连接模型…");
    $("layer-label").textContent = "L —";
    var stream = new EventSource("jspace/stream");
    events = stream;
    stream.onopen = function () {
      if (epoch !== modeVersion) return;
      connected = true;
      connection("实时连接已建立");
      if (toolsMode) {
        message("tool-status", "已连接，等待当前会话中的工具更新。");
        return;
      }
      message("live-status", "已连接。发送一条消息，或等待当前会话生成。");
      sendEnabled();
    };
    stream.onerror = function () {
      if (epoch !== modeVersion) return;
      connected = false;
      generationActive = false;
      stale = true;
      readout = null;
      showVectorWords([], "", "等待词汇信号。");
      connection("连接中断 · 正在重试", "error");
      if (toolsMode) {
        message("tool-status", "尚未收到工具事件。请连接专用会话并检查本机服务。");
        message("observed-text", "等待工具重新连接。");
        unknown("连接中断，等待工具信号。", "等待连接");
        return;
      }
      message("live-status", "未收到实时数据。若需要认证，请先连接会话；静态站请使用文字输入或演示。");
      unknown("连接中断，当前心情无法判断。", "等待连接");
      sendEnabled();
    };
    stream.onmessage = function (event) {
      if (epoch !== modeVersion) return;
      var ev;
      try { ev = JSON.parse(event.data); } catch (_) { return; }
      if (!ev || typeof ev !== "object") return;
      if (toolsMode) {
        if (ev.type === "meta" && ev.tool_updates_enabled === false) {
          connection("需要专用会话", "waiting");
          message("tool-status", "请启用服务认证，再用专用会话密钥连接。");
          return;
        }
        if (ev.type !== "mood_update" || typeof ev.id !== "string" ||
            typeof ev.source !== "string" || typeof ev.text !== "string") return;
        var fresh = !seenToolEvents.has(ev.id);
        seenToolEvents.add(ev.id);
        if (seenToolEvents.size > 64) seenToolEvents.delete(seenToolEvents.values().next().value);
        text = ev.text;
        raw("tool-label", ev.source);
        message("source-tag", "{source} · 文字", { source: ev.source });
        if (ev.phase === "interrupted") {
          message("tool-status", "显示中断前收到的回复片段。");
          connection("模型回复已中断", "error");
        } else {
          message("tool-status", ev.phase === "update" ? "显示工具最新推送的文字。" : "显示工具最近一次已完成的回复。");
          connection(ev.phase === "update" ? "模型正在回复 · 实时更新" : "工具回复 · 已完成");
        }
        var neighbors = Array.isArray(ev.neighbors) ? ev.neighbors : [];
        showVectorWords(neighbors.map(function (entry) {
          return entry && { word: entry.word, score: entry.cosine };
        }), neighbors.length ? "MiniLM · 余弦相似度" : "",
        neighbors.length ? "根据回复文字，在本机词表中找语义近邻。" : "当前推送没有文字嵌入词汇。");
        analyze(fresh, ev.ts * 1000);
        return;
      }
      if (ev.type === "meta") {
        demoReadout = ev.is_demo === true;
        if (typeof ev.model === "string") raw("model-label", ev.model);
        else message("model-label", "未知模型");
        $("layer-label").textContent = "L " + (Number.isInteger(ev.layer) ? ev.layer : "—");
        message("source-tag", ev.is_demo ? "实时模型 · DEMO 合成" : "实时模型");
      } else if (ev.type === "request_start") {
        if (typeof ev.request_id !== "string") return;
        activeRequest = ev.request_id;
        lastSeq = -1;
        generationActive = true;
        lastTokenAt = Date.now();
        stale = false;
        text = "";
        readout = null;
        showVectorWords([], "", "等待词汇信号。");
        message("observed-text", "模型开始生成…");
        unknown("新一轮生成开始，等待情绪信号。", "等待信号");
        message("live-status", "模型正在生成，信号逐 token 更新。");
        sendEnabled();
      } else if (ev.type === "token") {
        if (typeof ev.request_id !== "string" || !Number.isInteger(ev.seq) || ev.seq < 0 || typeof ev.token !== "string") return;
        // A late subscriber may receive a backlog without request_start.
        if (ev.request_id !== activeRequest) {
          activeRequest = ev.request_id;
          lastSeq = -1;
          text = "";
          readout = null;
        }
        if (ev.seq <= lastSeq) return;
        lastSeq = ev.seq;
        text = (text + (typeof ev.text === "string" ? ev.text : ev.token)).slice(-5000);
        readout = ev.readout;
        var ranked = ["zh", "en"].flatMap(function (lang) {
          return Array.isArray(readout?.[lang]) ? readout[lang] : [];
        }).filter(function (entry) {
          return entry && typeof entry.token === "string" && Number.isFinite(entry.rank);
        }).sort(function (a, b) { return a.rank - b.rank; });
        showVectorWords(ranked.map(function (entry) {
          return { word: entry.token, score: entry.score };
        }), demoReadout ? "J-lens · 合成演示" : "J-lens · 分数", "按当前词汇读出的排名显示。");
        generationActive = true;
        lastTokenAt = Date.now();
        stale = false;
        connection("正在接收实时信号");
        analyze(true);
        sendEnabled();
      } else if (ev.type === "request_end" && ev.request_id === activeRequest) {
        generationActive = false;
        if (ev.status !== "complete") {
          stale = true;
          readout = null;
          showVectorWords([], "", "等待词汇信号。");
          unknown("生成已中断，部分输出不作为完整结果。", "生成中断");
        }
        message("live-status", ev.status === "complete" ? "本轮生成已完成，当前显示本轮结果。" : "生成中断，请检查服务后重试。");
        connection(ev.status === "complete" ? "本轮结果 · 已完成" : "生成中断", ev.status === "complete" ? "ready" : "error");
        sendEnabled();
      }
    };
  }

  function setMode(next) {
    modeVersion++;
    if (events) events.close();
    if (timer) clearInterval(timer);
    if (requestController) requestController.abort();
    events = timer = requestController = null;
    mode = next;
    $("vector-section").hidden = mode !== "tools" && mode !== "live";
    showVectorWords([], "", "等待词汇信号。");
    seenToolEvents.clear();
    history = [];
    text = "";
    readout = null;
    connected = generationActive = stale = false;
    activeRequest = null;
    lastSeq = -1;
    drawHistory();
    lastUpdate = undefined;
    message("update-time", "尚未更新");
    document.querySelectorAll("[data-mode]").forEach(function (button) {
      var selected = button.dataset.mode === mode;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    ["text", "live", "tools", "demo"].forEach(function (name) { $(name + "-controls").hidden = name !== mode; });
    message("source-tag", { text: "文字输入", live: "实时模型", tools: "工具集成", demo: "演示 · 合成数据" }[mode]);
    sendEnabled();
    if (mode === "text") {
      connection("本地即时分析");
      text = $("mood-input").value;
      analyze(!!text);
    } else if (mode === "demo") {
      connection("演示中 · 非真实监测");
      demoPaused = false;
      demoIndex = 0;
      message("demo-toggle", "暂停演示 Ⅱ");
      demoStep();
      timer = setInterval(function () { if (!demoPaused) demoStep(); }, 3500);
    } else if (mode === "tools") {
      message("tool-label", "尚无工具更新");
      message("tool-status", "等待 Codex、Claude Code、OpenCode 或其他工具推送文字。");
      analyze(false);
      unknown("工具发来新文字后，这里会即时更新。");
      openLive();
    } else {
      analyze(false);
      unknown("连接后等待模型生成，不会使用其他来源的结果。");
      openLive();
      timer = setInterval(function () {
        if (generationActive && !stale && Date.now() - lastTokenAt > 15000) {
          stale = true;
          readout = null;
          showVectorWords([], "", "等待词汇信号。");
          unknown("15 秒未收到新 token，等待信号恢复。", "信号已暂停");
          connection("信号暂停", "waiting");
        }
      }, 1000);
    }
  }

  document.querySelectorAll("[data-mode]").forEach(function (button) { button.addEventListener("click", function () { setMode(button.dataset.mode); }); });
  $("mood-input").addEventListener("input", function () {
    $("char-count").textContent = $("mood-input").value.length + " / 2000";
    if (mode === "text") { text = $("mood-input").value; analyze(true); }
  });
  document.querySelectorAll("[data-example]").forEach(function (button) {
    button.addEventListener("click", function () {
      $("mood-input").value = I18n.language() === "en" ? button.dataset.exampleEn : button.dataset.example;
      $("mood-input").dispatchEvent(new Event("input"));
      $("mood-input").focus();
    });
  });
  $("live-basis").addEventListener("change", function () {
    history = [];
    drawHistory();
    lastUpdate = undefined;
    message("update-time", "尚未更新");
    analyze(false);
    if (!connected || stale) unknown("等待当前来源的新信号。", "等待信号");
    else analyze(!!text || !!readout);
  });
  $("demo-toggle").addEventListener("click", function () {
    demoPaused = !demoPaused;
    message("demo-toggle", demoPaused ? "继续演示 ▶" : "暂停演示 Ⅱ");
    connection(demoPaused ? "演示已暂停" : "演示中 · 非真实监测", demoPaused ? "waiting" : "ready");
  });
  $("reset").addEventListener("click", function () { history = []; drawHistory(); });
  $("prompt-form").addEventListener("submit", async function (event) {
    event.preventDefault();
    var prompt = $("live-prompt").value.trim();
    if (!prompt || !connected || requestController || generationActive) return;
    var epoch = modeVersion;
    var controller = new AbortController();
    requestController = controller;
    sendEnabled();
    message("live-status", "消息已发送，等待模型生成…");
    try {
      var response = await fetch("v1/chat/completions", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "user", content: prompt }], max_tokens: 64, stream: false }),
        signal: controller.signal,
      });
      var payload = await response.json();
      if (epoch !== modeVersion) return;
      if (!response.ok) throw new Error(typeof payload.error === "string" ? payload.error : "HTTP " + response.status);
    } catch (error) {
      if (epoch === modeVersion && error.name !== "AbortError") {
        stale = true;
        readout = null;
        message("live-status", "发送失败：{error}", { error: error.message });
        unknown("消息发送失败，等待新的有效生成。", "等待信号");
      }
    } finally {
      if (epoch === modeVersion) { requestController = null; sendEnabled(); }
    }
  });
  window.addEventListener("pagehide", function () {
    if (events) events.close();
    if (timer) clearInterval(timer);
    if (requestController) requestController.abort();
  });
  $("open-window").addEventListener("click", function (event) {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    var popup = window.open(this.href, "j7scope-mood", "popup,width=440,height=720");
    if (popup) {
      popup.opener = null;
      event.preventDefault();
    }
  });
  $("language-switch").addEventListener("click", function () {
    I18n.toggle();
    if (lastResult) render(lastResult, Object.assign({}, lastOptions, { record: false }));
    if (lastUpdate) message("update-time", "{time} 更新", { time: clock(lastUpdate) });
    drawHistory();
  });
  var query = new URLSearchParams(window.location.search);
  var compact = query.get("view") === "compact";
  document.body.classList.toggle("compact", compact);
  $("compact-view").hidden = compact;
  $("open-window").hidden = !compact;
  $("full-dashboard").hidden = !compact;
  var initial = query.get("mode");
  setMode(["text", "live", "tools", "demo"].includes(initial) ? initial : compact ? "tools" : "text");
})();
