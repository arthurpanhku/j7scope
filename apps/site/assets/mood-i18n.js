(function () {
  "use strict";
  var english = {
    "J7Scope · 心情实时看板": "J7Scope · Live Mood Dashboard",
    "即时查看文字和模型实时读出中的正负情绪信号。": "See positive and negative signals in text and live model readouts.",
    "主导航": "Main navigation",
    "实验回放": "Trace gallery",
    "实时读出 ↗": "Live readout ↗",
    "集成小窗 ↗": "Compact window ↗",
    "另开窗口 ↗": "Open separate window ↗",
    "完整看板 ↗": "Full dashboard ↗",
    "此刻，心情怎么样？": "How does it feel right now?",
    "让文字里的情绪，变得一目了然。": "See the tone in the words, at a glance.",
    "本地即时分析": "Instant local analysis",
    "好心情": "Positive mood",
    "坏心情": "Negative mood",
    "心情有点复杂": "Mixed mood",
    "等待信号": "Waiting for a signal",
    "此刻，有一点晴朗": "A little brighter",
    "此刻，有一点阴天": "A little overcast",
    "晴与阴，都在这里": "A bit of both",
    "等一句话，感受一下": "Start with a few words",
    "观察到了偏积极的情绪信号。": "The words suggest a positive tone.",
    "观察到了偏消极的情绪信号。": "The words suggest a negative tone.",
    "正负信号接近，暂不判断好坏。": "Positive and negative signals are close.",
    "没有足够的情绪关键词，暂不判断。": "No clear mood signal in the words yet.",
    "输入文字后，这里会即时更新。": "This updates as you type.",
    "情绪指数 —": "Mood index —",
    "情绪指数 {score}": "Mood index {score}",
    "根据文字中的情绪关键词判断，不代表真实心理状态。": "Based on keywords in the text. This does not measure an actual mental state.",
    "给心情一个入口": "Choose a signal source",
    "信号来源": "Signal source",
    "文字输入": "Text input",
    "实时模型": "Live model",
    "工具集成": "Agent tools",
    "正在连接本地小窗…": "Pairing the local dashboard…",
    "配对失败，请重新运行接入命令。": "Pairing failed. Run the integration command again.",
    "体验演示": "Demo",
    "写下此刻的一句话": "Write a sentence",
    "比如：今天的事情很顺利，我很开心。": "For example: Things went well today. I feel happy.",
    "仅在浏览器内分析，不上传": "Analyzed locally in your browser",
    "试一句": "Try one",
    "☀ 开心的一天": "☀ A happy day",
    "☁ 有点低落": "☁ Feeling low",
    "等待连接模型…": "Waiting for the model…",
    "分析依据": "Analyze",
    "模型输出文字 · 最近 500 字符": "Model output · last 500 characters",
    "J-lens 词汇读出 · 探索性": "J-lens vocabulary · experimental",
    "发送消息，观察实时变化": "Send a message to watch the signal change",
    "请用一句话描述开心的一天。": "Describe a happy day in one sentence.",
    "发送消息": "Send message",
    "发送 ↗": "Send ↗",
    "连接当前服务的实时事件流…": "Connecting to the local event stream…",
    "需要认证？连接会话 →": "Authentication required? Connect →",
    "合成示例，展示界面效果": "Synthetic examples",
    "跟着几段不同的心情，看看表情、指数和趋势如何一起变化。": "Watch the face, index and trend change across example moods.",
    "暂停演示 Ⅱ": "Pause demo Ⅱ",
    "继续演示 ▶": "Resume demo ▶",
    "尚无工具更新": "No agent updates yet",
    "等待 Codex、Claude Code、OpenCode 或其他工具推送文字。": "Waiting for text from Codex, Claude Code, OpenCode or another tool.",
    "连接专用会话 →": "Connect a dedicated session →",
    "看到了哪些信号": "Matched signals",
    "向量空间词汇": "Vector-space words",
    "MiniLM · 余弦相似度": "MiniLM · cosine similarity",
    "根据回复文字，在本机词表中找语义近邻。": "Semantic neighbors of the reply text in a local vocabulary.",
    "等待词汇信号。": "Waiting for vocabulary signals.",
    "当前推送没有文字嵌入词汇。": "This update has no text embedding neighbors.",
    "J-lens · 分数": "J-lens · score",
    "J-lens · 合成演示": "J-lens · synthetic demo",
    "按当前词汇读出的排名显示。": "Ranked by the current vocabulary readout.",
    "模型正在回复 · 实时更新": "Agent reply streaming",
    "模型回复已中断": "Agent reply interrupted",
    "显示中断前收到的回复片段。": "Showing the reply fragment received before interruption.",
    "0 个关键词": "0 keyword matches",
    "{count} 个关键词": "{count} keyword matches",
    "还没有可判断的情绪词。": "No mood keywords matched yet.",
    "正在观察的文字": "Observed text",
    "尚未更新": "No updates yet",
    "你的文字会出现在这里。": "Your text appears here.",
    "心情变化趋势": "Mood trend",
    "心情的起伏": "Mood over time",
    "最近 32 次更新 · 当前来源": "Last 32 updates · current source",
    "清空记录 ↺": "Clear history ↺",
    "暂无情绪变化记录": "No mood history yet",
    "有了信号，就会留下轨迹。": "Signals will appear here as they arrive.",
    "较早": "Earlier",
    "刚刚": "Now",
    "这些结果如何得到？": "How is this calculated?",
    "使用中英情绪关键词与简单否定规则，不是训练过的情绪分类器。指数是正负关键词权重之差除以总权重，不是概率。没有命中或正负混合时不会强制判断好坏。反讽、引用和复杂语境可能误判。J-lens 模式仅匹配当前 top-k 中的完整情绪词，按原始排名加权；孤立词不能反映否定语境，也不能证明模型具有真实情绪。":
      "This rule matches English and Chinese mood keywords and simple negation. The index is the difference between positive and negative weights divided by their total. It is not a probability. Missing and mixed signals are left unclassified. Sarcasm, quotations and context can lead to errors. The J-lens view weights whole words by their top-k rank; isolated words cannot establish negation or a model's actual feelings.",
    "{time} 更新": "Updated {time}",
    "最近 {count} 次变化：{states}": "Last {count} updates: {states}",
    "暂时没有文字。": "No text yet.",
    "同期输出文字 · 不参与当前判断": "Model output · not used for this signal",
    "按 J-lens top-k 情绪词排名加权；探索性信号，不代表模型真实情绪。": "Words are weighted by J-lens top-k rank. This is an exploratory signal.",
    "正在连接…": "Connecting…",
    "等待当前服务的实时事件流…": "Waiting for the local event stream…",
    "实时连接已建立": "Live connection ready",
    "已连接，等待当前会话中的工具更新。": "Connected. Waiting for agent updates in this session.",
    "已连接。发送一条消息，或等待当前会话生成。": "Connected. Send a message or wait for generation in this session.",
    "连接中断 · 正在重试": "Disconnected · retrying",
    "尚未收到工具事件。请连接专用会话并检查本机服务。": "No tool events. Connect a dedicated session and check the local service.",
    "等待工具重新连接。": "Waiting for the tool stream to reconnect.",
    "连接中断，等待工具信号。": "Disconnected. Waiting for an agent signal.",
    "等待连接": "Waiting for connection",
    "未收到实时数据。若需要认证，请先连接会话；静态站请使用文字输入或演示。": "No live data. Connect a session if required, or use Text input or Demo.",
    "连接中断，当前心情无法判断。": "Disconnected. No current mood signal.",
    "需要专用会话": "Dedicated session required",
    "请启用服务认证，再用专用会话密钥连接。": "Enable service authentication and connect with a dedicated session key.",
    "{source} · 文字": "{source} · text",
    "显示工具最新推送的文字。": "Showing the latest text from the tool.",
    "显示工具最近一次已完成的回复。": "Showing the tool's latest completed reply.",
    "工具更新 · 最新一次": "Latest agent update",
    "工具回复 · 已完成": "Agent reply complete",
    "未知模型": "Unknown model",
    "实时模型 · DEMO 合成": "Live model · synthetic demo",
    "模型开始生成…": "The model started generating…",
    "新一轮生成开始，等待情绪信号。": "New reply started. Waiting for a mood signal.",
    "模型正在生成，信号逐 token 更新。": "Generating. Signals update with each token.",
    "正在接收实时信号": "Receiving live signals",
    "生成已中断，部分输出不作为完整结果。": "Generation was interrupted. The partial reply is not a complete result.",
    "生成中断": "Generation interrupted",
    "本轮生成已完成，当前显示本轮结果。": "Generation complete. Showing the result for this reply.",
    "生成中断，请检查服务后重试。": "Generation interrupted. Check the service and try again.",
    "本轮结果 · 已完成": "Reply complete",
    "演示 · 合成数据": "Demo · synthetic",
    "演示中 · 非真实监测": "Demo running · synthetic",
    "工具发来新文字后，这里会即时更新。": "This updates as tools send text.",
    "连接后等待模型生成，不会使用其他来源的结果。": "Connect and wait for the model to generate a reply.",
    "15 秒未收到新 token，等待信号恢复。": "No new tokens for 15 seconds. Waiting for the stream.",
    "信号已暂停": "Signal paused",
    "信号暂停": "Signal paused",
    "等待当前来源的新信号。": "Waiting for a new signal from this source.",
    "演示已暂停": "Demo paused",
    "消息已发送，等待模型生成…": "Message sent. Waiting for the model…",
    "发送失败：{error}": "Send failed: {error}",
    "消息发送失败，等待新的有效生成。": "The message could not be sent. Waiting for the next reply."
  };

  var query = new URLSearchParams(location.search);
  var stored;
  try { stored = localStorage.getItem("j7scope-mood-language"); } catch (_) {}
  var language = query.get("lang") || stored;
  language = language === "en" ? "en" : "zh";
  var bindings = new Map();
  var staticCopy = [];
  var walker = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    var node = walker.currentNode;
    if (Object.hasOwn(english, node.nodeValue)) staticCopy.push({ node: node, key: node.nodeValue });
  }
  document.querySelectorAll("[aria-label], [placeholder], meta[name=description]").forEach(function (node) {
    ["aria-label", "placeholder", "content"].forEach(function (attr) {
      var key = node.getAttribute(attr);
      if (Object.hasOwn(english, key)) staticCopy.push({ node: node, key: key, attr: attr });
    });
  });

  function t(key, values) {
    var copy = language === "en" && Object.hasOwn(english, key) ? english[key] : key;
    return copy.replace(/\{(\w+)\}/g, function (match, name) { return values && Object.hasOwn(values, name) ? values[name] : match; });
  }

  function message(id, key, values) {
    bindings.set(id, { key: key, values: values });
    document.getElementById(id).textContent = t(key, values);
  }

  function raw(id, value) {
    bindings.delete(id);
    document.getElementById(id).textContent = value;
  }

  function apply() {
    document.documentElement.lang = language === "en" ? "en" : "zh-CN";
    staticCopy.forEach(function (entry) {
      if (!entry.node.isConnected) return;
      if (entry.attr) entry.node.setAttribute(entry.attr, t(entry.key));
      else entry.node.nodeValue = t(entry.key);
    });
    bindings.forEach(function (entry, id) { message(id, entry.key, entry.values); });
    ["compact-view", "open-window", "full-dashboard"].forEach(function (id) {
      var link = document.getElementById(id);
      var params = new URL(link.href, location.href);
      params.searchParams.set("lang", language);
      link.setAttribute("href", "mood.html?" + params.searchParams.toString());
    });
    var button = document.getElementById("language-switch");
    button.textContent = language === "en" ? "中文" : "EN";
    button.setAttribute("aria-label", language === "en" ? "Switch to Chinese" : "Switch to English");
  }

  window.MoodI18n = {
    t: t, message: message, raw: raw, language: function () { return language; },
    toggle: function () {
      language = language === "en" ? "zh" : "en";
      try { localStorage.setItem("j7scope-mood-language", language); } catch (_) {}
      try {
        var url = new URL(location.href);
        url.searchParams.set("lang", language);
        history.replaceState(null, "", url);
      } catch (_) {}
      apply();
    }
  };
  apply();
})();
