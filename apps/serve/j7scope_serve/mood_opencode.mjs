// OpenCode V1 project plugin. Forward only the completed assistant text.
export const J7ScopeMood = async ({ client }) => {
  const pending = new Map();
  const sent = new Set();
  return {
    event: async ({ event }) => {
      const key = process.env.J7SCOPE_MOOD_SESSION_KEY;
      const port = Number(process.env.J7SCOPE_MOOD_PORT || 8799);
      if (!key || !Number.isInteger(port) || port < 1 || port > 65535) return;
      const props = event.properties;
      if (event.type === "message.updated") {
        const info = props?.info;
        if (info?.role === "assistant" && typeof info.sessionID === "string"
            && typeof info.id === "string" && !sent.has(info.id)) {
          pending.delete(info.sessionID);
          pending.set(info.sessionID, info.id);
          if (pending.size > 64) pending.delete(pending.keys().next().value);
        }
        return;
      }
      if (event.type !== "session.idle") return;
      const id = props?.sessionID;
      const messageID = pending.get(id);
      if (!messageID) return;
      pending.delete(id);
      sent.add(messageID);
      if (sent.size > 128) sent.delete(sent.values().next().value);
      try {
        const { data } = await client.session.message({ path: { id, messageID }, signal: AbortSignal.timeout(1000) });
        if (data?.info?.role !== "assistant" || data.info.id !== messageID || data.info.error) return;
        const parts = data.parts?.filter(part => part.type === "text" && !part.synthetic && !part.ignored
                                                 && typeof part.text === "string");
        const text = Array.from(parts?.map(part => part.text).join("\n").trim() || "").slice(-500).join("").trim();
        if (!text) return;
        await fetch(`http://127.0.0.1:${port}/mood/events`, {
          method: "POST", redirect: "error", signal: AbortSignal.timeout(1000),
          headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
          body: JSON.stringify({ source: "OpenCode", text, phase: "complete" }),
        });
      } catch {
        // An unavailable dashboard should not interrupt OpenCode.
      }
    },
  };
};
