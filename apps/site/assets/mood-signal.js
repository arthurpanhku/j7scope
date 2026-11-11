/* Small, inspectable vocabulary heuristic. Values are indices, not probabilities. */
(function (root) {
  "use strict";
  var positive = ["开心", "快乐", "高兴", "愉快", "幸福", "喜悦", "满意", "放松", "安心", "期待", "惊喜", "顺利", "喜欢", "感谢", "兴奋", "平静", "happy", "happiness", "joy", "joyful", "glad", "delighted", "excited", "grateful", "relaxed", "calm", "hopeful", "cheerful", "love"];
  var negative = ["难过", "伤心", "沮丧", "焦虑", "烦躁", "愤怒", "生气", "失望", "痛苦", "害怕", "疲惫", "孤独", "悲伤", "讨厌", "担心", "难受", "紧张", "糟糕", "委屈", "sad", "sadness", "unhappy", "angry", "anger", "anxious", "anxiety", "upset", "disappointed", "miserable", "afraid", "tired", "lonely", "stressed", "worried", "hate", "terrible", "frustrated"];
  var lexicon = new Map();
  positive.forEach(function (word) { lexicon.set(word, 1); });
  negative.forEach(function (word) { lexicon.set(word, -1); });
  var words = Array.from(lexicon.keys()).sort(function (a, b) { return b.length - a.length; });
  var chinese = words.filter(function (word) { return /[\u3400-\u9fff]/.test(word); });
  var english = words.filter(function (word) { return /^[a-z]+$/.test(word); });
  var pattern = chinese.join("|") + "|\\b(?:" + english.join("|") + ")\\b";

  function summarize(evidence) {
    var pos = 0, neg = 0;
    evidence.forEach(function (entry) {
      if (entry.polarity > 0) pos += entry.weight;
      else neg += entry.weight;
    });
    var total = pos + neg;
    var score = total ? (pos - neg) / total : null;
    return {
      state: score === null ? "unknown" : score > 0.2 ? "good" : score < -0.2 ? "bad" : "mixed",
      score: score,
      positive: evidence.filter(function (entry) { return entry.polarity > 0; }).length,
      negative: evidence.filter(function (entry) { return entry.polarity < 0; }).length,
      evidence: evidence,
    };
  }

  function classifyText(input) {
    var text = typeof input === "string" ? input : "";
    var matcher = new RegExp(pattern, "gi"), evidence = [], match;
    while ((match = matcher.exec(text)) !== null) {
      var word = match[0].toLowerCase();
      var polarity = lexicon.get(word);
      var prefix = text.slice(Math.max(0, match.index - 45), match.index);
      var negation = /[\u3400-\u9fff]/.test(word)
        ? prefix.match(/(?:(?:并不|不是|没有|并非|不|没|未)(?:很|太|怎么|那么|非常|特别|再|一点|也|觉得|感觉|感到|能|会|够|的|到|得|想|是|真|就|有|多|么)*)+$/)
        : prefix.match(/\b(?:not|never|hardly|no|don['’]t|isn['’]t|wasn['’]t|didn['’]t)(?:\s+(?:very|really|so|at\s+all|particularly|feel|feeling)){0,2}\s+$/i);
      var flips = negation ? ((negation[0].match(/并不|不是|没有|并非|不|没|未/g) || []).length || 1) : 0;
      if (flips % 2) polarity *= -1;
      evidence.push({ text: (negation ? negation[0] : "") + match[0], polarity: polarity, weight: 1 });
    }
    return summarize(evidence);
  }

  function classifyReadout(readout) {
    var evidence = [];
    if (!readout || typeof readout !== "object") return summarize(evidence);
    ["zh", "en", "other"].forEach(function (bucket) {
      if (!Array.isArray(readout[bucket])) return;
      readout[bucket].forEach(function (entry, index) {
        if (!entry || typeof entry.token !== "string" || !Number.isFinite(entry.score)) return;
        var word = entry.token.replace(/^[▁Ġ]+/, "").trim().toLowerCase();
        if (!lexicon.has(word)) return;
        var rank = Number.isInteger(entry.rank) && entry.rank >= 0 ? entry.rank : index;
        evidence.push({ text: entry.token.trim(), polarity: lexicon.get(word), weight: 1 / (rank + 1) });
      });
    });
    return summarize(evidence);
  }

  var api = { classifyText: classifyText, classifyReadout: classifyReadout };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MoodSignal = api;
})(typeof window !== "undefined" ? window : globalThis);
