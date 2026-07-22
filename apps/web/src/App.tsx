import { Fragment, useEffect, useMemo, useState } from "react";
import type { FormEvent, KeyboardEvent, ReactNode } from "react";
import Plot from "react-plotly.js";
import {
  Activity,
  ArrowRightLeft,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  Cpu,
  FlaskConical,
  Gauge,
  HardDrive,
  KeyRound,
  Languages,
  Layers3,
  MessageSquare,
  Network,
  RotateCcw,
  Save,
  Search,
  Send,
  Server,
  Settings,
  SlidersHorizontal,
  Table2,
  UserRound,
  X,
  XCircle,
} from "lucide-react";
import { listRuns, loadRun } from "./data";
import type {
  Language,
  PatchRecord,
  ProjectionPoint,
  ReadoutRecord,
  RunData,
  RunIndexEntry,
  TokenScore,
} from "./types";

type ViewKey = "agent" | "matrix" | "map" | "pairs" | "layers";
type ProviderId = "openai" | "anthropic" | "openrouter" | "deepseek" | "local";

interface LlmSettings {
  provider: ProviderId;
  model: string;
  endpoint: string;
  apiKey: string;
  temperature: number;
  maxTokens: number;
  stream: boolean;
  rememberKey: boolean;
}

interface ChatMessage {
  id: string;
  role: "agent" | "user";
  text: string;
  timestamp: string;
}

const languageLabel: Record<Language, string> = {
  zh: "中文",
  en: "English",
};

const categoryLabel: Record<string, string> = {
  deception: "Deception",
  manipulation: "Manipulation",
  concession: "Concession",
  eval_awareness: "Evaluation",
  emotion: "Emotion",
  multihop: "Multihop",
  entity: "Entity",
};

const viewItems: Array<{ key: ViewKey; label: string; icon: typeof Table2 }> = [
  { key: "agent", label: "Agent", icon: MessageSquare },
  { key: "matrix", label: "Patching", icon: Table2 },
  { key: "pairs", label: "Pairs", icon: Search },
  { key: "map", label: "J-Space", icon: Network },
  { key: "layers", label: "Layers", icon: Layers3 },
];

const providerOptions: Array<{
  id: ProviderId;
  label: string;
  endpoint: string;
  models: string[];
}> = [
  {
    id: "openai",
    label: "OpenAI",
    endpoint: "https://api.openai.com/v1",
    models: ["gpt-5", "gpt-4.1", "o3"],
  },
  {
    id: "anthropic",
    label: "Anthropic",
    endpoint: "https://api.anthropic.com",
    models: ["claude-sonnet-4", "claude-opus-4", "claude-3-7-sonnet"],
  },
  {
    id: "openrouter",
    label: "OpenRouter",
    endpoint: "https://openrouter.ai/api/v1",
    models: ["openai/gpt-5", "anthropic/claude-sonnet-4", "qwen/qwen3-235b-a22b"],
  },
  {
    id: "deepseek",
    label: "DeepSeek",
    endpoint: "https://api.deepseek.com",
    models: ["deepseek-chat", "deepseek-reasoner"],
  },
  {
    id: "local",
    label: "Local",
    endpoint: "http://127.0.0.1:11434/v1",
    models: ["qwen2.5:7b", "llama3.1:8b", "custom-local-model"],
  },
];

const defaultSettings: LlmSettings = {
  provider: "openai",
  model: "gpt-5",
  endpoint: "https://api.openai.com/v1",
  apiKey: "",
  temperature: 0.2,
  maxTokens: 1200,
  stream: true,
  rememberKey: false,
};

const settingsStorageKey = "j7scope.llmSettings";

function formatPercent(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) return "n/a";
  return `${Math.round(value * 100)}%`;
}

function loadSettings(): LlmSettings {
  if (typeof window === "undefined") return defaultSettings;
  const raw = window.localStorage.getItem(settingsStorageKey);
  if (!raw) return defaultSettings;
  try {
    return { ...defaultSettings, ...JSON.parse(raw) } as LlmSettings;
  } catch {
    return defaultSettings;
  }
}

function persistSettings(settings: LlmSettings) {
  const stored = {
    ...settings,
    apiKey: settings.rememberKey ? settings.apiKey : "",
  };
  window.localStorage.setItem(settingsStorageKey, JSON.stringify(stored));
}

function nowLabel() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

function buildAgentReply({
  prompt,
  data,
  category,
  selectedLayer,
  selectedPatch,
  settings,
}: {
  prompt: string;
  data: RunData;
  category: string;
  selectedLayer: number;
  selectedPatch?: PatchRecord;
  settings: LlmSettings;
}) {
  const lower = prompt.toLowerCase();
  const summary = data.metrics.summary;
  const scope = category === "all" ? "全部概念类别" : categoryLabel[category] ?? category;
  const bestLayer = data.layerScan.rows.reduce((best, row) =>
    row.cross_language_success > best.cross_language_success ? row : best,
  );

  if (lower.includes("layer") || prompt.includes("层")) {
    return `我会先看 layer scan。当前 run 里跨语言迁移最高的是 L${bestLayer.layer}，成功率 ${formatPercent(bestLayer.cross_language_success)}，null 是 ${formatPercent(bestLayer.null_success)}。如果是真实实验，我会把这个层作为 patching 主候选，再检查 source-language leakage 有没有同步升高。`;
  }

  if (lower.includes("provider") || lower.includes("api") || prompt.includes("模型")) {
    return `当前设置里的 provider 是 ${settings.provider}，模型是 ${settings.model}。这一版 Agent 面板先走本地 artifact 解释逻辑，不会把 API key 发出去；下一步可以把这里接到 Provider，让它基于当前 J-Space 状态生成更完整的实验解读。`;
  }

  if (lower.includes("patch") || prompt.includes("迁移") || prompt.includes("因果")) {
    const patch = selectedPatch
      ? `${languageLabel[selectedPatch.source_language]} → ${languageLabel[selectedPatch.target_language]}，${selectedPatch.category}，L${selectedPatch.patch_layer}，concept score ${selectedPatch.concept_score.toFixed(3)}`
      : "还没有选中的 patch";
    return `我会把因果证据放在第一位。当前选中 patch 是：${patch}。run 级别 cross-language transport 是 ${formatPercent(summary.cross_language_success)}，null 是 ${formatPercent(summary.null_success)}。如果这个 gap 在真实 run 中稳定存在，就比单纯 CKA 更像论文主证据。`;
  }

  return `我现在看到的是 ${data.manifest.label}，J-lens L${data.manifest.jlens_layer}，patch layer L${selectedLayer}，范围是${scope}。当前 cross-language transport ${formatPercent(summary.cross_language_success)}，language preserved ${formatPercent(summary.language_preservation)}。我建议右侧先盯三件事：J-Space 聚类是否按概念靠近、patching matrix 是否高于 null、layer scan 是否有清晰峰值。`;
}

function buildAgentSystemPrompt({
  data,
  category,
  selectedLayer,
  selectedPatch,
}: {
  data: RunData;
  category: string;
  selectedLayer: number;
  selectedPatch?: PatchRecord;
}) {
  const summary = data.metrics.summary;
  const scope = category === "all" ? "all categories" : category;
  const patch = selectedPatch
    ? {
        pair_id: selectedPatch.pair_id,
        source_language: selectedPatch.source_language,
        target_language: selectedPatch.target_language,
        category: selectedPatch.category,
        patch_layer: selectedPatch.patch_layer,
        transport_success: selectedPatch.transport_success,
        concept_score: selectedPatch.concept_score,
        null_gap: selectedPatch.null_gap,
        leakage: selectedPatch.source_language_leakage,
      }
    : null;

  return [
    "You are the J7Scope research agent.",
    "Answer in concise Chinese unless the user asks otherwise.",
    "Ground your answer in the current J-Space experiment artifacts.",
    "Be explicit when a result is demo/synthetic rather than a real experiment.",
    "",
    `Run: ${data.manifest.label}`,
    `Model under study: ${data.manifest.model}`,
    `J-lens layer: ${data.manifest.jlens_layer}`,
    `Selected patch layer: ${selectedLayer}`,
    `Scope: ${scope}`,
    `Cross-language transport: ${formatPercent(summary.cross_language_success)}`,
    `Null success: ${formatPercent(summary.null_success)}`,
    `Language preservation: ${formatPercent(summary.language_preservation)}`,
    `Selected patch: ${JSON.stringify(patch)}`,
  ].join("\n");
}

async function callLlmProvider({
  prompt,
  messages,
  data,
  category,
  selectedLayer,
  selectedPatch,
  settings,
}: {
  prompt: string;
  messages: ChatMessage[];
  data: RunData;
  category: string;
  selectedLayer: number;
  selectedPatch?: PatchRecord;
  settings: LlmSettings;
}) {
  if (!settings.apiKey.trim()) {
    throw new Error("Missing API key");
  }

  const apiUrl =
    settings.provider === "deepseek"
      ? "/api/deepseek/chat/completions"
      : `${settings.endpoint.replace(/\/$/, "")}/chat/completions`;

  const recentMessages = messages.slice(-8).map((message) => ({
    role: message.role === "agent" ? "assistant" : "user",
    content: message.text,
  }));

  const response = await fetch(apiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${settings.apiKey.trim()}`,
    },
    body: JSON.stringify({
      model: settings.model,
      messages: [
        {
          role: "system",
          content: buildAgentSystemPrompt({ data, category, selectedLayer, selectedPatch }),
        },
        ...recentMessages,
        { role: "user", content: prompt },
      ],
      temperature: settings.temperature,
      max_tokens: settings.maxTokens,
      stream: false,
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText}${body ? `: ${body.slice(0, 240)}` : ""}`);
  }

  const payload = await response.json();
  const content = payload?.choices?.[0]?.message?.content;
  if (typeof content !== "string" || !content.trim()) {
    throw new Error("Provider returned an empty response");
  }
  return content.trim();
}

function App() {
  const [runs, setRuns] = useState<RunIndexEntry[]>([]);
  const [activeRun, setActiveRun] = useState<RunIndexEntry | null>(null);
  const [data, setData] = useState<RunData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<ViewKey>("agent");
  const [category, setCategory] = useState("all");
  const [selectedLayer, setSelectedLayer] = useState(12);
  const [selectedPatchId, setSelectedPatchId] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [llmSettings, setLlmSettings] = useState<LlmSettings>(loadSettings);
  const [settingsSaved, setSettingsSaved] = useState(false);
  const [chatDraft, setChatDraft] = useState("");
  const [chatPending, setChatPending] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "agent",
      text: "我可以帮你读当前 run 的 J-Space、patching matrix 和 layer scan。问我某个概念是否共享、哪个层值得看，或者让我们一起设计下一轮实验。",
      timestamp: "ready",
    },
  ]);

  useEffect(() => {
    listRuns()
      .then((entries) => {
        setRuns(entries);
        setActiveRun(entries[0] ?? null);
      })
      .catch((error: Error) => setLoadError(error.message));
  }, []);

  useEffect(() => {
    if (!activeRun) return;
    setData(null);
    loadRun(activeRun)
      .then((run) => {
        setData(run);
        setLoadError(null);
      })
      .catch((error: Error) => setLoadError(error.message));
  }, [activeRun]);

  useEffect(() => {
    if (!data) return;
    setSelectedLayer(data.metrics.primary_patch_layer ?? data.manifest.patch_layers[0] ?? 12);
  }, [data?.manifest.run_id]);

  const filteredPatches = useMemo(() => {
    if (!data) return [];
    return data.patches.filter((patch) => {
      const categoryMatch = category === "all" || patch.category === category;
      return categoryMatch && patch.patch_layer === selectedLayer;
    });
  }, [category, data, selectedLayer]);

  useEffect(() => {
    if (!filteredPatches.length) {
      setSelectedPatchId(null);
      return;
    }
    if (!selectedPatchId || !filteredPatches.some((patch) => patch.patch_id === selectedPatchId)) {
      setSelectedPatchId(filteredPatches[0].patch_id);
    }
  }, [filteredPatches, selectedPatchId]);

  const selectedPatch = useMemo(
    () => filteredPatches.find((patch) => patch.patch_id === selectedPatchId) ?? filteredPatches[0],
    [filteredPatches, selectedPatchId],
  );

  const categories = data?.manifest.categories ?? [];
  const activeProvider = providerOptions.find((provider) => provider.id === llmSettings.provider) ?? providerOptions[0];

  const saveSettings = () => {
    persistSettings(llmSettings);
    setSettingsSaved(true);
    window.setTimeout(() => setSettingsSaved(false), 1400);
  };

  const resetSettings = () => {
    window.localStorage.removeItem(settingsStorageKey);
    setLlmSettings(defaultSettings);
    setSettingsSaved(false);
  };

  const sendChatMessage = async () => {
    const text = chatDraft.trim();
    if (!text || !data || chatPending) return;
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      text,
      timestamp: nowLabel(),
    };
    const previousMessages = chatMessages;
    setChatMessages((messages) => [...messages, userMessage]);
    setChatDraft("");
    setChatPending(true);

    let replyText = "";
    try {
      replyText = await callLlmProvider({
        prompt: text,
        messages: previousMessages,
        data,
        category,
        selectedLayer,
        selectedPatch,
        settings: llmSettings,
      });
    } catch (error) {
      const fallback = buildAgentReply({
        prompt: text,
        data,
        category,
        selectedLayer,
        selectedPatch,
        settings: llmSettings,
      });
      replyText = `Provider call failed: ${error instanceof Error ? error.message : String(error)}\n\nLocal fallback:\n${fallback}`;
    } finally {
      setChatPending(false);
    }

    const agentMessage: ChatMessage = {
      id: `agent-${Date.now()}`,
      role: "agent",
      text: replyText,
      timestamp: nowLabel(),
    };
    setChatMessages((messages) => [...messages, agentMessage]);
  };

  if (loadError) {
    return (
      <main className="app-shell">
        <div className="empty-state">
          <XCircle size={24} />
          <strong>Artifact load failed</strong>
          <span>{loadError}</span>
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="app-shell">
        <div className="empty-state">
          <Activity size={24} />
          <strong>Loading J7Scope artifacts</strong>
        </div>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <img src="/logo.svg" alt="J7Scope" />
          <div>
            <h1>J7Scope Explorer</h1>
            <span>{data.manifest.model}</span>
          </div>
        </div>
        <div className="run-controls">
          <div className="provider-status">
            <Server size={15} />
            <span>{activeProvider.label}</span>
            <strong>{llmSettings.model}</strong>
          </div>
          <label>
            Run
            <select
              value={activeRun?.run_id ?? ""}
              onChange={(event) => setActiveRun(runs.find((run) => run.run_id === event.target.value) ?? null)}
            >
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Patch layer
            <select value={selectedLayer} onChange={(event) => setSelectedLayer(Number(event.target.value))}>
              {data.manifest.patch_layers.map((layer) => (
                <option key={layer} value={layer}>
                  L{layer}
                </option>
              ))}
            </select>
          </label>
          {data.manifest.is_demo ? <span className="demo-badge">demo</span> : null}
          <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="Open settings">
            <Settings size={18} />
          </button>
        </div>
      </header>

      <section className="summary-strip">
        <Metric icon={Cpu} label="J-lens" value={`L${data.manifest.jlens_layer}`} detail={data.manifest.status} />
        <Metric
          icon={ArrowRightLeft}
          label="Cross-language"
          value={formatPercent(data.metrics.summary.cross_language_success)}
          detail="concept transport"
        />
        <Metric
          icon={Gauge}
          label="Null"
          value={formatPercent(data.metrics.summary.null_success)}
          detail="random / unrelated"
        />
        <Metric
          icon={Languages}
          label="Language kept"
          value={formatPercent(data.metrics.summary.language_preservation)}
          detail="target surface"
        />
      </section>

      <section className="workspace">
        <aside className="sidebar">
          <div className="control-group">
            <span className="control-title">View</span>
            <div className="tab-list">
              {viewItems.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    className={activeView === item.key ? "tab active" : "tab"}
                    key={item.key}
                    onClick={() => setActiveView(item.key)}
                  >
                    <Icon size={16} />
                    {item.label}
                  </button>
                );
              })}
            </div>
          </div>
          <div className="control-group">
            <span className="control-title">Category</span>
            <div className="category-list">
              <button className={category === "all" ? "category active" : "category"} onClick={() => setCategory("all")}>
                All
              </button>
              {categories.map((name) => (
                <button
                  className={category === name ? "category active" : "category"}
                  key={name}
                  onClick={() => setCategory(name)}
                >
                  {categoryLabel[name] ?? name}
                </button>
              ))}
            </div>
          </div>
        </aside>

        <section className="main-panel">
          {activeView === "agent" ? (
            <AgentWorkspace
              data={data}
              category={category}
              selectedLayer={selectedLayer}
              selectedPatch={selectedPatch}
              messages={chatMessages}
              draft={chatDraft}
              pending={chatPending}
              settings={llmSettings}
              onDraftChange={setChatDraft}
              onSend={sendChatMessage}
            />
          ) : null}
          {activeView === "matrix" ? (
            <PatchingMatrix
              categories={categories}
              patches={filteredPatches}
              selectedPatchId={selectedPatch?.patch_id}
              onSelectPatch={setSelectedPatchId}
            />
          ) : null}
          {activeView === "pairs" ? (
            <PairExplorer
              readouts={data.readouts}
              patches={filteredPatches}
              selectedPatch={selectedPatch}
              onSelectPatch={setSelectedPatchId}
            />
          ) : null}
          {activeView === "map" ? <JSpaceMap data={data} category={category} /> : null}
          {activeView === "layers" ? <LayerScan data={data} /> : null}
        </section>
      </section>
      <SettingsDrawer
        open={settingsOpen}
        provider={activeProvider}
        settings={llmSettings}
        saved={settingsSaved}
        onClose={() => setSettingsOpen(false)}
        onReset={resetSettings}
        onSave={saveSettings}
        onChange={setLlmSettings}
      />
    </main>
  );
}

function SettingsDrawer({
  open,
  provider,
  settings,
  saved,
  onClose,
  onReset,
  onSave,
  onChange,
}: {
  open: boolean;
  provider: (typeof providerOptions)[number];
  settings: LlmSettings;
  saved: boolean;
  onClose: () => void;
  onReset: () => void;
  onSave: () => void;
  onChange: (settings: LlmSettings) => void;
}) {
  const [openSections, setOpenSections] = useState({
    provider: true,
    generation: true,
    storage: false,
  });

  const update = (patch: Partial<LlmSettings>) => onChange({ ...settings, ...patch });

  const chooseProvider = (providerId: ProviderId) => {
    const next = providerOptions.find((option) => option.id === providerId) ?? providerOptions[0];
    update({
      provider: next.id,
      endpoint: next.endpoint,
      model: next.models[0],
    });
  };

  return (
    <>
      <button className={open ? "settings-backdrop visible" : "settings-backdrop"} onClick={onClose} aria-label="Close settings" />
      <aside className={open ? "settings-drawer open" : "settings-drawer"} aria-hidden={!open}>
        <div className="settings-header">
          <div>
            <span>Settings</span>
            <h2>LLM Provider</h2>
          </div>
          <button className="icon-button quiet" onClick={onClose} aria-label="Close settings">
            <X size={18} />
          </button>
        </div>

        <div className="settings-provider-card">
          <Server size={18} />
          <div>
            <span>{provider.label}</span>
            <strong>{settings.model}</strong>
          </div>
        </div>

        <div className="settings-content">
          <SettingsSection
            icon={Server}
            title="Provider"
            open={openSections.provider}
            onToggle={() => setOpenSections((state) => ({ ...state, provider: !state.provider }))}
          >
            <div className="provider-grid">
              {providerOptions.map((option) => (
                <button
                  className={settings.provider === option.id ? "provider-option active" : "provider-option"}
                  key={option.id}
                  onClick={() => chooseProvider(option.id)}
                >
                  <span>{option.label}</span>
                  {settings.provider === option.id ? <Check size={15} /> : null}
                </button>
              ))}
            </div>
            <Field label="Model">
              <input
                list="llm-models"
                value={settings.model}
                onChange={(event) => update({ model: event.target.value })}
                placeholder="model"
              />
              <datalist id="llm-models">
                {provider.models.map((model) => (
                  <option key={model} value={model} />
                ))}
              </datalist>
            </Field>
            <Field label="Endpoint">
              <input value={settings.endpoint} onChange={(event) => update({ endpoint: event.target.value })} placeholder="https://..." />
            </Field>
            <Field label="API key">
              <div className="key-field">
                <KeyRound size={15} />
                <input
                  type="password"
                  value={settings.apiKey}
                  onChange={(event) => update({ apiKey: event.target.value })}
                  placeholder="sk-..."
                />
              </div>
            </Field>
          </SettingsSection>

          <SettingsSection
            icon={SlidersHorizontal}
            title="Generation"
            open={openSections.generation}
            onToggle={() => setOpenSections((state) => ({ ...state, generation: !state.generation }))}
          >
            <Field label={`Temperature ${settings.temperature.toFixed(1)}`}>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={settings.temperature}
                onChange={(event) => update({ temperature: Number(event.target.value) })}
              />
            </Field>
            <Field label="Max tokens">
              <input
                type="number"
                min="64"
                max="8192"
                step="64"
                value={settings.maxTokens}
                onChange={(event) => update({ maxTokens: Number(event.target.value) })}
              />
            </Field>
            <Toggle label="Stream responses" checked={settings.stream} onChange={(stream) => update({ stream })} />
          </SettingsSection>

          <SettingsSection
            icon={HardDrive}
            title="Storage"
            open={openSections.storage}
            onToggle={() => setOpenSections((state) => ({ ...state, storage: !state.storage }))}
          >
            <Toggle label="Remember API key" checked={settings.rememberKey} onChange={(rememberKey) => update({ rememberKey })} />
            <div className="settings-note">Local browser storage</div>
          </SettingsSection>
        </div>

        <div className="settings-actions">
          <button className="secondary-button" onClick={onReset}>
            <RotateCcw size={16} />
            Reset
          </button>
          <button className={saved ? "primary-button saved" : "primary-button"} onClick={onSave}>
            {saved ? <Check size={16} /> : <Save size={16} />}
            {saved ? "Saved" : "Save"}
          </button>
        </div>
      </aside>
    </>
  );
}

function SettingsSection({
  icon: Icon,
  title,
  open,
  onToggle,
  children,
}: {
  icon: typeof Server;
  title: string;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  return (
    <section className={open ? "settings-section open" : "settings-section"}>
      <button className="settings-section-toggle" onClick={onToggle}>
        <span>
          <Icon size={16} />
          {title}
        </span>
        <ChevronDown size={16} />
      </button>
      <div className="settings-section-body">{children}</div>
    </section>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="settings-field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <button className={checked ? "toggle-row active" : "toggle-row"} onClick={() => onChange(!checked)}>
      <span>{label}</span>
      <span className="switch" aria-hidden="true">
        <span />
      </span>
    </button>
  );
}

function Metric({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="metric">
      <Icon size={18} />
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </div>
  );
}

function AgentWorkspace({
  data,
  category,
  selectedLayer,
  selectedPatch,
  messages,
  draft,
  pending,
  settings,
  onDraftChange,
  onSend,
}: {
  data: RunData;
  category: string;
  selectedLayer: number;
  selectedPatch?: PatchRecord;
  messages: ChatMessage[];
  draft: string;
  pending: boolean;
  settings: LlmSettings;
  onDraftChange: (value: string) => void;
  onSend: () => void;
}) {
  const activeProvider = providerOptions.find((provider) => provider.id === settings.provider) ?? providerOptions[0];
  const summary = data.metrics.summary;
  const selectedScope = category === "all" ? "All categories" : categoryLabel[category] ?? category;

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    onSend();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  };

  return (
    <div className="surface agent-workspace">
      <section className="agent-chat-panel">
        <div className="agent-panel-heading">
          <div>
            <h2>Research Agent</h2>
            <p>{activeProvider.label} · {settings.model}</p>
          </div>
          <Bot size={22} />
        </div>

        <div className="chat-thread">
          {messages.map((message) => (
            <div className={message.role === "agent" ? "chat-message agent" : "chat-message user"} key={message.id}>
              <div className="message-avatar">{message.role === "agent" ? <Bot size={16} /> : <UserRound size={16} />}</div>
              <div className="message-body">
                <div className="message-meta">
                  <strong>{message.role === "agent" ? "Agent" : "You"}</strong>
                  <span>{message.timestamp}</span>
                </div>
            <p>{message.text}</p>
          </div>
        </div>
      ))}
          {pending ? (
            <div className="chat-message agent">
              <div className="message-avatar">
                <Bot size={16} />
              </div>
              <div className="message-body pending-message">
                <div className="message-meta">
                  <strong>Agent</strong>
                  <span>calling provider</span>
                </div>
                <p>正在读取当前 J-Space context 并请求 LLM...</p>
              </div>
            </div>
          ) : null}
        </div>

        <form className="chat-composer" onSubmit={handleSubmit}>
          <textarea
            value={draft}
            onChange={(event) => onDraftChange(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about cross-language transport, layer choice, null gap..."
            rows={3}
            disabled={pending}
          />
          <button className="send-button" disabled={!draft.trim() || pending} type="submit">
            <Send size={17} />
            {pending ? "Calling" : "Send"}
          </button>
        </form>
      </section>

      <section className="agent-evidence-panel">
        <div className="agent-panel-heading">
          <div>
            <h2>J-Space Context</h2>
            <p>{selectedScope} · patch L{selectedLayer}</p>
          </div>
          <Network size={22} />
        </div>

        <div className="evidence-strip">
          <EvidenceStat label="Transport" value={formatPercent(summary.cross_language_success)} />
          <EvidenceStat label="Null" value={formatPercent(summary.null_success)} />
          <EvidenceStat label="Language" value={formatPercent(summary.language_preservation)} />
        </div>

        <MiniJSpacePlot data={data} category={category} />

        <div className="patch-brief">
          <span className="mini-title">Selected Patch</span>
          {selectedPatch ? (
            <>
              <div className="patch-brief-title">
                <strong>
                  {languageLabel[selectedPatch.source_language]} → {languageLabel[selectedPatch.target_language]}
                </strong>
                <StatusPill ok={selectedPatch.transport_success} label={selectedPatch.transport_success ? "transport" : "miss"} />
              </div>
              <div className="patch-brief-grid">
                <Score label="Concept" value={selectedPatch.concept_score} />
                <Score label="Null gap" value={selectedPatch.null_gap} />
                <Score label="Leakage" value={selectedPatch.source_language_leakage} />
              </div>
              <TokenPills tokens={selectedPatch.readout.slice(0, 6)} />
            </>
          ) : (
            <p className="muted-copy">Select a patching cell to bind the agent to a concrete intervention.</p>
          )}
        </div>
      </section>
    </div>
  );
}

function EvidenceStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="evidence-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MiniJSpacePlot({ data, category }: { data: RunData; category: string }) {
  const traces = useMemo(() => {
    const points = data.projections.points.filter((point) => category === "all" || point.category === category);
    const byId = new Map(points.map((point) => [point.id, point]));
    const lineX: Array<number | null> = [];
    const lineY: Array<number | null> = [];

    data.projections.links.forEach((link) => {
      const source = byId.get(link.source);
      const target = byId.get(link.target);
      if (!source || !target) return;
      lineX.push(source.x, target.x, null);
      lineY.push(source.y, target.y, null);
    });

    const zh = points.filter((point) => point.language === "zh");
    const en = points.filter((point) => point.language === "en");

    return [
      {
        type: "scatter",
        mode: "lines",
        x: lineX,
        y: lineY,
        line: { color: "rgba(88, 99, 113, 0.18)", width: 1 },
        hoverinfo: "skip",
        name: "pairs",
      },
      {
        type: "scatter",
        mode: "markers",
        x: zh.map((point) => point.x),
        y: zh.map((point) => point.y),
        text: zh.map((point) => `${point.pair_id}<br>${point.concept}<br>中文`),
        hovertemplate: "%{text}<extra></extra>",
        marker: { color: "#2563eb", size: 8, line: { color: "#fff", width: 1 } },
        name: "zh",
      },
      {
        type: "scatter",
        mode: "markers",
        x: en.map((point) => point.x),
        y: en.map((point) => point.y),
        text: en.map((point) => `${point.pair_id}<br>${point.concept}<br>English`),
        hovertemplate: "%{text}<extra></extra>",
        marker: { color: "#dc5a3a", size: 8, line: { color: "#fff", width: 1 } },
        name: "en",
      },
    ];
  }, [category, data]);

  return (
    <div className="mini-plot-frame">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 286,
          margin: { l: 28, r: 12, t: 8, b: 28 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(255,255,255,0.72)",
          xaxis: { zeroline: false, showgrid: true, title: "" },
          yaxis: { zeroline: false, showgrid: true, title: "" },
          showlegend: false,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}

function PatchingMatrix({
  categories,
  patches,
  selectedPatchId,
  onSelectPatch,
}: {
  categories: string[];
  patches: PatchRecord[];
  selectedPatchId?: string;
  onSelectPatch: (patchId: string) => void;
}) {
  const columns = [
    {
      key: "zh-en",
      label: "中文 → English",
      match: (patch: PatchRecord) =>
        patch.control_type === "cross_language_concept" &&
        patch.source_language === "zh" &&
        patch.target_language === "en",
    },
    {
      key: "en-zh",
      label: "English → 中文",
      match: (patch: PatchRecord) =>
        patch.control_type === "cross_language_concept" &&
        patch.source_language === "en" &&
        patch.target_language === "zh",
    },
    {
      key: "same",
      label: "same language",
      match: (patch: PatchRecord) => patch.control_type === "same_language_concept",
    },
    {
      key: "random",
      label: "random norm",
      match: (patch: PatchRecord) => patch.control_type === "random_same_norm",
    },
    {
      key: "unrelated",
      label: "unrelated",
      match: (patch: PatchRecord) => patch.control_type === "unrelated_concept",
    },
  ];

  const visibleCategories = categories.filter((name) => patches.some((patch) => patch.category === name));

  return (
    <div className="surface">
      <div className="surface-heading">
        <div>
          <h2>Cross-Lingual Patching Matrix</h2>
          <p>Success rate by concept family and control condition.</p>
        </div>
        <FlaskConical size={22} />
      </div>
      <div className="matrix-grid" style={{ gridTemplateColumns: `minmax(136px, 1.1fr) repeat(${columns.length}, minmax(112px, 1fr))` }}>
        <div className="matrix-header">category</div>
        {columns.map((column) => (
          <div className="matrix-header" key={column.key}>
            {column.label}
          </div>
        ))}
        {visibleCategories.map((name) => (
          <Fragment key={name}>
            <div className="matrix-row-label" key={`${name}-label`}>
              <strong>{categoryLabel[name] ?? name}</strong>
              <span>{name}</span>
            </div>
            {columns.map((column) => {
              const cellRows = patches.filter((patch) => patch.category === name && column.match(patch));
              const successRate = cellRows.length
                ? cellRows.filter((patch) => patch.transport_success).length / cellRows.length
                : 0;
              const selected = cellRows.some((patch) => patch.patch_id === selectedPatchId);
              const alpha = Math.max(0.08, Math.min(0.86, successRate));
              const firstPatch = cellRows[0];
              return (
                <button
                  className={selected ? "matrix-cell selected" : "matrix-cell"}
                  disabled={!firstPatch}
                  key={`${name}-${column.key}`}
                  onClick={() => firstPatch && onSelectPatch(firstPatch.patch_id)}
                  style={{
                    background: `rgba(23, 123, 116, ${alpha})`,
                    color: successRate > 0.58 ? "#fff" : "#20312f",
                  }}
                >
                  <strong>{cellRows.length ? formatPercent(successRate) : "—"}</strong>
                  <span>{cellRows.length ? `${cellRows.length} patches` : "empty"}</span>
                </button>
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function PairExplorer({
  readouts,
  patches,
  selectedPatch,
  onSelectPatch,
}: {
  readouts: ReadoutRecord[];
  patches: PatchRecord[];
  selectedPatch?: PatchRecord;
  onSelectPatch: (patchId: string) => void;
}) {
  const pairId = selectedPatch?.pair_id ?? patches[0]?.pair_id;
  const zh = readouts.find((row) => row.pair_id === pairId && row.language === "zh");
  const en = readouts.find((row) => row.pair_id === pairId && row.language === "en");
  const auditRows = patches.slice(0, 24);

  return (
    <div className="surface">
      <div className="surface-heading">
        <div>
          <h2>Pair Explorer</h2>
          <p>{pairId ? `${pairId} · ${selectedPatch?.category ?? ""}` : "No patch selected"}</p>
        </div>
        <Search size={22} />
      </div>

      <div className="pair-layout">
        <div className="patch-list">
          {auditRows.map((patch) => (
            <button
              className={patch.patch_id === selectedPatch?.patch_id ? "patch-item active" : "patch-item"}
              key={patch.patch_id}
              onClick={() => onSelectPatch(patch.patch_id)}
            >
              <span>
                {languageLabel[patch.source_language]} → {languageLabel[patch.target_language]}
              </span>
              <strong>{categoryLabel[patch.category] ?? patch.category}</strong>
              <small>{patch.control_type.replace(/_/g, " ")}</small>
            </button>
          ))}
        </div>

        <div className="pair-main">
          <div className="prompt-grid">
            {zh ? <PromptPane title="中文 probe" readout={zh} /> : null}
            {en ? <PromptPane title="English probe" readout={en} /> : null}
          </div>

          {selectedPatch ? (
            <div className="patch-detail">
              <div className="patch-title">
                <div>
                  <h3>
                    {languageLabel[selectedPatch.source_language]} → {languageLabel[selectedPatch.target_language]} · L
                    {selectedPatch.patch_layer}
                  </h3>
                  <span>{selectedPatch.control_type.replace(/_/g, " ")}</span>
                </div>
                <StatusPill ok={selectedPatch.transport_success} label={selectedPatch.transport_success ? "transport" : "miss"} />
                <StatusPill ok={selectedPatch.language_preserved} label="language" />
              </div>

              <div className="score-row">
                <Score label="Concept score" value={selectedPatch.concept_score} />
                <Score label="Null gap" value={selectedPatch.null_gap} />
                <Score label="Leakage" value={selectedPatch.source_language_leakage} />
              </div>

              <div className="readout-columns">
                <div>
                  <span className="mini-title">Patched J-lens readout</span>
                  <TokenPills tokens={selectedPatch.readout} />
                </div>
                <div>
                  <span className="mini-title">Actual next-token channel</span>
                  <TokenPills tokens={selectedPatch.next_token} />
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function PromptPane({ title, readout }: { title: string; readout: ReadoutRecord }) {
  return (
    <div className="prompt-pane">
      <div className="pane-heading">
        <Languages size={16} />
        <span>{title}</span>
      </div>
      <p>{readout.prompt}</p>
      <TokenPills tokens={readout.topk} />
    </div>
  );
}

function TokenPills({ tokens }: { tokens: TokenScore[] }) {
  return (
    <div className="token-list">
      {tokens.map((token) => (
        <span className={token.is_expected ? "token expected" : "token"} key={`${token.rank}-${token.token}`}>
          {token.token}
          <small>{token.score.toFixed(1)}</small>
        </span>
      ))}
    </div>
  );
}

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={ok ? "status ok" : "status bad"}>
      {ok ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
      {label}
    </span>
  );
}

function Score({ label, value }: { label: string; value: number }) {
  return (
    <div className="score">
      <span>{label}</span>
      <strong>{value.toFixed(3)}</strong>
    </div>
  );
}

function JSpaceMap({ data, category }: { data: RunData; category: string }) {
  const traces = useMemo(() => {
    const points = data.projections.points.filter((point) => category === "all" || point.category === category);
    const byId = new Map(points.map((point) => [point.id, point]));
    const lineX: Array<number | null> = [];
    const lineY: Array<number | null> = [];

    data.projections.links.forEach((link) => {
      const source = byId.get(link.source);
      const target = byId.get(link.target);
      if (!source || !target) return;
      lineX.push(source.x, target.x, null);
      lineY.push(source.y, target.y, null);
    });

    const groups = new Map<string, ProjectionPoint[]>();
    points.forEach((point) => {
      const key = `${point.language}:${point.condition}`;
      groups.set(key, [...(groups.get(key) ?? []), point]);
    });

    const markerColor: Record<string, string> = {
      "zh:baseline": "#2563eb",
      "en:baseline": "#dc5a3a",
      "zh:patched": "#23a2a4",
      "en:patched": "#b7791f",
    };

    return [
      {
        type: "scatter",
        mode: "lines",
        x: lineX,
        y: lineY,
        line: { color: "rgba(88, 99, 113, 0.25)", width: 1 },
        hoverinfo: "skip",
        name: "zh/en pair",
      },
      ...Array.from(groups.entries()).map(([key, group]) => ({
        type: "scatter",
        mode: "markers",
        x: group.map((point) => point.x),
        y: group.map((point) => point.y),
        text: group.map((point) => `${point.pair_id}<br>${point.concept}<br>${languageLabel[point.language]}`),
        hovertemplate: "%{text}<extra></extra>",
        marker: {
          color: markerColor[key],
          size: group.map((point) => (point.condition === "patched" ? 11 : 9)),
          symbol: group.map((point) => (point.condition === "patched" ? "diamond" : "circle")),
          line: { color: "#ffffff", width: 1.4 },
        },
        name: key.replace(":", " · "),
      })),
    ];
  }, [category, data]);

  return (
    <div className="surface">
      <div className="surface-heading">
        <div>
          <h2>J-Space Map</h2>
          <p>{data.projections.basis}</p>
        </div>
        <Network size={22} />
      </div>
      <div className="plot-frame">
        <Plot
          data={traces}
          layout={{
            autosize: true,
            height: 540,
            margin: { l: 36, r: 20, t: 20, b: 36 },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(255,255,255,0.72)",
            xaxis: { zeroline: false, title: "projection 1" },
            yaxis: { zeroline: false, title: "projection 2" },
            legend: { orientation: "h", y: 1.08 },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </div>
    </div>
  );
}

function LayerScan({ data }: { data: RunData }) {
  const rows = data.layerScan.rows;
  const x = rows.map((row) => row.layer);
  const traces = [
    {
      type: "scatter",
      mode: "lines+markers",
      x,
      y: rows.map((row) => row.cross_language_success),
      name: "cross-language abstract",
      line: { color: "#177b74", width: 3 },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      x,
      y: rows.map((row) => row.concrete_success),
      name: "concrete",
      line: { color: "#7256a5", width: 3 },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      x,
      y: rows.map((row) => row.null_success),
      name: "null",
      line: { color: "#9b4a34", width: 3, dash: "dot" },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      x,
      y: rows.map((row) => row.source_language_leakage),
      name: "language leakage",
      line: { color: "#d49b2a", width: 3 },
    },
  ];

  return (
    <div className="surface">
      <div className="surface-heading">
        <div>
          <h2>Layer Scan</h2>
          <p>{data.layerScan.metric}</p>
        </div>
        <Layers3 size={22} />
      </div>
      <div className="plot-frame">
        <Plot
          data={traces}
          layout={{
            autosize: true,
            height: 520,
            margin: { l: 46, r: 24, t: 20, b: 42 },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(255,255,255,0.72)",
            xaxis: { title: "patch layer", dtick: 2 },
            yaxis: { title: "rate", range: [0, 1] },
            legend: { orientation: "h", y: 1.08 },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </div>
    </div>
  );
}

export default App;
