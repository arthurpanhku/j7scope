# J7Scope 平台开发计划：Replay-first J-Space

> 目标：让 J7Scope 成为顶尖实验室、大学与开发者**愿意引用**的项目。
> 本文是显示层与平台的详细开发计划，与研究主线（README §4–§6 的 M1–M3）并行推进。

## 0. 定位（三个前提决策）

| 维度 | 决策 | 含义 |
|---|---|---|
| 引用身份 | **公开演示平台** | 任何人打开就能玩的 J-Space playground，靠可见度与传播带动引用 |
| 算力资源 | **基本无 GPU** | 平台本身不能依赖常驻 GPU；GPU 只在"采集"时按小时租 |
| 显示严谨度 | **分层严谨** | 默认视图直观好看，每个信号可展开到 null 基线 / 对照 / 置信区间 |

这三条组合起来有一个显式矛盾：**"公开演示平台"通常需要常驻 GPU 做实时推理，而我们"基本无 GPU"。** 本计划的全部设计都围绕如何化解它。

## 1. 核心策略：把「采集」与「显示」解耦

**GPU 只在采集时需要（按小时租）；平台本身零 GPU 成本。** 显示层不直接连模型，而是回放**录制好的 trace**。

这不是妥协，而是升级 —— 它同时更省钱、也更容易被引用：

- live 会话转瞬即逝，**没人能引用一个已经关掉的推理进程**；trace 是永久、可复现、可深链的 artifact —— 论文引用的从来是后者。
- Neuronpedia 等被广泛引用的可解释性平台，本质也是"预计算 + 浏览"，而非实时推理。我们做同一件事，对象是"时间轴上的 J-space"。
- 静态站（GitHub Pages）免费、无限并发、永不宕机；实时模式仍保留（有卡即可用），但平台不依赖它。

```
        ┌─────────── 采集（按小时租 GPU / Colab / 社区）───────────┐
        │  j7scope_serve --record   →   trace/<id>/ (版本化 artifact) │
        └──────────────────────────────┬───────────────────────────┘
                                        │ 上传 gallery / Zenodo(DOI)
        ┌───────────── 显示（零 GPU，GitHub Pages 静态站）─────────────┐
        │  Replay Viewer：时序回放 · 深链 · 双 trace 对齐 · null 带 · 导出 │
        └──────────────────────────────────────────────────────────────┘
```

## 2. 架构：四个构件

### 构件 A — Trace 格式（被引用的核心资产）

一个版本化 schema，一次会话 = 一组 artifact 文件。关键在于**严谨层在采集时就烤进文件**（采集时算几乎免费，显示时才能"点开即有对照"）。沿用 [`j7scope/artifacts.py`](../j7scope/artifacts.py) 的 JSON/JSONL + `sort_keys` + `ensure_ascii=False` 约定。见 [§3 Trace Schema v1](#3-trace-schema-v1规范)。

### 构件 B — Replay Viewer（平台本体）

现有 [`apps/serve/viewer/index.html`](../apps/serve/viewer/index.html) 与 [`apps/web`](../apps/web) 升级：

- **时序回放**：按原始 token 时间轴重放，保留"看模型思考"的 live 质感（可暂停 / 逐帧 / 变速）。
- **深链到时刻**：URL 形如 `#trace=deception-01&token=42`，论文脚注与社媒可精确指向"就这个 token，两列同时点亮"。这是传播的最小原子单位。
- **双 trace 并排对齐**：同一概念的 zh / en 两条 trace 按位置对齐回放 —— 研究主张的"证据画面"。
- **null 带**：每个 bar 后画一条淡色分布带（shuffled-pair 基线范围），bar 超出 null 带才算真信号，一眼诚实。
- **导出**：任意画面一键导出论文级 SVG/PNG + 底层数据 JSON + BibTeX。

### 构件 C — 采集工具包（让社区成为你的 GPU）

- sidecar 加 `--record` 旗标：任何有卡的人跑一次即产出标准 trace，可 PR 进 gallery。
- **Colab 免费层笔记本**：T4 跑 Qwen2.5-1.5B-Instruct，"10 分钟采你自己的 trace，无需本地 GPU" —— 开发者入口。
- 官方按小时租卡采"精选集"（欺骗/操纵/评估意识各若干条 + 一条完整 opencode 编码会话）。

### 构件 D — Harness 集成的新定位

opencode / codex 集成从"必须实时"变为"实时可选、录制常态"：接开源模型的人边写码边录 trace，会话结束得到一个可分享的回放链接。实时模式仍在（README §2），但平台不依赖它。

## 3. Trace Schema v1（规范）

一次会话写入 `traces/<trace_id>/`：

```
traces/<trace_id>/
├── manifest.json     # 会话元数据 + 溯源 + schema 版本
├── tokens.jsonl      # 每个生成 token 一行：readout + 严谨层
├── metrics.json      # 会话级摘要
└── align.json        # 可选：与另一条 trace 的位置对齐（双语并排用）
```

### 3.1 `manifest.json`

```jsonc
{
  "schema_version": 1,
  "trace_id": "deception-zh-01",
  "kind": "single | parallel_member",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "revision": "<hf commit hash>",       // 精确复现
  "layer": 18,
  "language": "zh",
  "concept": "deception",
  "prompt": "…",
  "jacobian": {                          // J_l 溯源：可复现的前提
    "corpus_id": "generic-v1",
    "n_prompts": 8, "n_probes": 16, "seed": 0,
    "sha1": "<J_l 张量哈希>"
  },
  "capture": {                           // 采集环境
    "tool": "j7scope_serve --record",
    "tool_version": "0.2.0",
    "device": "A10G", "dtype": "bf16",
    "created_at": "2026-07-22T…Z"
  },
  "is_demo": false,                      // mock/合成数据一律 true
  "doi": null,                           // 发 Zenodo 后回填
  "parallel_group": "deception-01"       // 同组 zh/en trace 共享，用于配对
}
```

### 3.2 `tokens.jsonl`（每行一个 token）

```jsonc
{
  "seq": 42,
  "ts_rel": 1.734,                       // 相对会话起点秒数，回放用
  "token": "欺骗",
  "token_script": "zh",
  "readout": {                           // 与 live 协议一致（protocol.py）
    "zh": [{"token":"欺骗","score":11.8,"rank":0}, …],
    "en": [{"token":"deception","score":11.0,"rank":1}, …],
    "other": []
  },
  "rigor": {                             // ★ 采集时烤入的严谨层
    "null": {                            // shuffled-pair 基线的分布摘要
      "metric": "topk_overlap",
      "mean": 0.21, "p05": 0.08, "p95": 0.37, "n_shuffles": 200
    },
    "same_lang_baseline": 0.62,          // 同语言不同 prompt 的重叠（对照上界）
    "cross_lang_overlap": 0.44,          // 本 token 跨语言重叠（观测值）
    "sharedness": {                      // 归一化到 null 的共享分
      "value": 0.71, "ci95": [0.55, 0.86],
      "definition": "(obs - null.mean) / (same_lang - null.mean)"
    }
  }
}
```

> **严谨层的语义**（对应 README §3.4 的统计陷阱）：显示层永远不裸报 `cross_lang_overlap`。默认视图画观测 bar + null 带；点开显示 `sharedness` 及其 CI，且注明"n≈30、d≈3584 下裸 CKA 基线可高达 ~0.7，必须对 null 归一化"。`sharedness` 的精确定义与 [`j7scope/metrics.py`](../j7scope/metrics.py) 单一实现共享，禁止在前端重算。

### 3.3 兼容性与稳定性

- `schema_version` 单调递增；viewer 读旧版本时降级显示，缺字段不崩。
- Trace 一旦发 DOI 即冻结；修订走新 `trace_id`（`…-v2`）。
- live 协议（[`protocol.py`](../apps/serve/j7scope_serve/protocol.py)）的 `token` 事件是 `tokens.jsonl` 一行的**流式子集**，二者共用 readout 结构，避免两套格式漂移。

## 4. 里程碑（按引用杠杆排序）

平台里程碑记作 **P1–P5**，与研究里程碑 M1–M3 区分。

### P1 · Trace schema v1 + `--record` + Replay 模式 —— 现在就能做（mock 可验证）

- **目标**：本地零 GPU 走通"录制 → 回放"全链路，mock 数据即可开发。
- **交付物**
  - `j7scope/trace.py`：schema v1 读写（复用 `artifacts.py` 的原子写）+ 校验器。
  - `j7scope/rigor.py`：`sharedness`、null 分布、CI 的单一实现（`metrics.py` 之上）。
  - `apps/serve/j7scope_serve` 加 `--record <dir>`：live 生成时把每 token 落盘为 trace；mock 后端合成 `rigor` 字段（标 `is_demo`）。
  - `experiments/build_demo_trace.py`：从探针语料生成一组明确标 demo 的 trace（含一对 zh/en 平行）。
  - viewer 增加 `?trace=<path>` 回放模式（读静态文件，不连 sidecar）+ 播放控制。
- **验收**：`build_demo_trace.py` 产出 trace → 浏览器打开回放，逐 token 点亮，每 bar 有 null 带；schema 校验器通过；无需 GPU。

### P2 · 静态 Gallery 站 + 深链 + 导出（GitHub Pages）

- **目标**：一个免费、永久在线的公开站点。
- **交付物**
  - `apps/web` 增加 gallery：读 `traces/index.json`，列出所有 trace（模型/概念/语言/是否 demo/是否有 DOI）。
  - 深链路由 `#trace=…&token=…`；双 trace 并排对齐视图（读 `align.json`）。
  - 导出：当前画面 → SVG/PNG + 数据 JSON + BibTeX 片段。
  - GitHub Actions：push 即构建并部署到 Pages；trace 作为静态资源随站发布。
- **验收**：Pages 上线，任何人可打开、深链可复现具体画面、导出的 SVG 可直接进论文。

### P3 · 首批真实 trace + Zenodo DOI（合并一次租卡，顺带 M1 首批数据）

- **目标**：把平台从"demo"升级为"有真数据可引用"。
- **交付物**
  - 按小时租 24GB 卡，用 `--record` 采官方精选集：欺骗/操纵/评估意识/情绪/多跳/实体各若干条，中英平行；外加一条完整 opencode 编码会话的 trace。
  - 发布到 Zenodo 取 DOI；仓库加 `CITATION.cff` 与 `datasets/README.md`（数据卡）。
  - **同一次租卡顺带跑 M1**：CKA/SVCCA + top-k 重叠 + null 基线，产出研究侧第一批数字（README §9 的 M1 勾选项）。
- **验收**：gallery 出现带 DOI 的真实 trace；`CITATION.cff` 可被 GitHub "Cite this repository" 识别；M1 首批相关性数字入 `results/`。

> **CPU 小模型预演结论（2026-07，[`experiments/capture_small.py`](../experiments/capture_small.py)）。**
> 在本机 CPU 上用 Qwen2.5-0.5B-Instruct 真跑通了整条 `hf` 流水线（并借此修掉了
> [`fitting.py`](../j7scope/fitting.py) `_Capture` 对新版 transformers 张量输出的一个真实 bug）。
> **但读出是噪声** —— 即便在 J 拟合的 in-distribution cloze 位置，top-k 也是无意义
> 的碎 token（`WaitForSeconds`、`.ToDecimal`、`闹`…）。原因是：(1) 随机 Jacobian 估计
> 严重欠采样（512 个样本估 896×896 矩阵，需要数千以上），(2) 0.5B 的 workspace 本就弱。
> **含义：小模型 + CPU 可行预算下没有信号，这不是平台问题，正是 P3 需要 GPU + 更大模型的实证理由。**
> **P3 前置动作**：先按 `estimate_jacobian` 的 TODO，用 `torch.func.jacrev` 精确 Jacobian
> 和上游 `anthropics/jacobian-lens` 在一个已知案例上校验拟合数学，确认 7B 跑出来的是信号而非噪声，再租卡。

### P4 · Colab 采集笔记本 + 社区提交流程

- **目标**：把 GPU 成本外包给社区，形成 trace 供给。
- **交付物**
  - `notebooks/capture_colab.ipynb`：T4 免费层跑 Qwen2.5-1.5B-Instruct，10 分钟采一条自定义 trace。
  - `CONTRIBUTING.md` + trace 提交模板 + CI 校验（schema、`is_demo` 标注、`sharedness` 未在前端重算）。
- **验收**：外部贡献者能在 Colab 免费层产出通过 CI 的 trace 并合入 gallery。

### P5 · 严谨层完全体（并排对齐 + null 带 + 可展开审计）

- **目标**：兑现"分层严谨"——研究者第一眼就信任。
- **交付物**
  - 每个信号可展开：观测值、null 分布、same-lang 上界、`sharedness` + CI 同屏。
  - 双语并排对齐回放的"共享/分歧"高亮（超出 null 带才高亮）。
  - "方法学"页：解释 `sharedness` 定义、n≪d 陷阱、为何不裸报 CKA/overlap。
- **验收**：随机抽一个高亮信号，能在 3 次点击内看到它相对 null 的完整证据链。

### 依赖关系

```
P1 ──▶ P2 ──▶ P3 ──▶ P4
        └──────────▶ P5
```

P1 是一切的地基（schema）。P2 让它可见。P3 让它可引用。P4/P5 分别扩供给与提严谨度，可并行。

## 5. 采集经济学（无 GPU 前提下的可持续性）

| 来源 | 成本 | 用途 |
|---|---|---|
| 按小时租 24GB 卡 | 用时计费，几美元/批 | 官方精选集、M1 数据（P3） |
| Colab 免费 T4 | 0 | 社区自采、小模型 demo（P4） |
| 社区有卡贡献者 | 0（外包） | 长尾概念、跨模型（P4） |
| 静态托管 GitHub Pages | 0 | 平台常驻显示 |

**关键点**：平台的常驻成本是 0；花钱的只有"采集"，且按需、可批处理、可外包。

## 6. 引用基础设施（让引用真的发生）

- **Zenodo DOI**：每批 trace 一个 DOI —— 论文的落点。
- **`CITATION.cff`**：GitHub 原生"Cite this repository"。
- **深链 + 导出的 SVG**：降低"在论文里用你的图"的摩擦到近乎零。
- **数据卡（datasets/README）**：模型、层、J_l 溯源、语料、严谨层定义 —— 审稿人要看的东西一次给全。
- **方法学页**：把 README §3 的统计陷阱做成平台一等公民，主动展示局限 = 可信度。

## 7. 风险与诚实边界

- **平台是放大器，不是结论。** 顶尖实验室最终引用的是 M1–M3 的**发现**；平台带来可见度，但没有真数据就只是漂亮的 demo。P3 把"采真实 trace"和"跑 M1"合并正是为此。
- **`sharedness` 是一个定义，不是真理。** 必须在方法学页写清它的假设与替代定义（CKA vs overlap vs patching），否则会被质疑是 cherry-picked 指标。
- **按字符集分 zh/en 是显示启发式**，不是语言学判定；混合 token、专名、代码会落到 `other`。文档需明说，避免过度解读。
- **schema 一旦被引用就难改。** v1 要尽量把溯源字段（model revision、J_l 哈希、seed）留全；宁可先冗余。

## 8. 与研究主线的关系

平台（P1–P5）与研究（M1–M3）不是两个项目，而是同一份数据的两种出口：

- 采集一次 trace，既喂 replay 显示，也喂 M1 的 CKA/SVCCA 计算。
- `rigor.sharedness` 与 `j7scope/metrics.py` 共享实现 —— 显示层与论文用**同一个数**。
- P3 的租卡批次同时产出"可引用的 trace"和"M1 首批结论"。

---

## English summary

**Replay-first J-Space platform.** Given the three constraints — public demo platform, essentially no GPU, layered rigor — the design decouples **capture** (needs a GPU, rented by the hour) from **display** (zero-GPU, static GitHub Pages). The display never runs a model; it replays recorded **traces**. Traces are permanent, reproducible, deep-linkable artifacts — which is what papers actually cite, unlike an ephemeral live session.

Four pieces: (A) a versioned **trace schema** that bakes the rigor layer (shuffled-pair null, same-language baseline, `sharedness` + CI) in at capture time; (B) a **replay viewer** with time-ordered playback, deep links to a specific token, side-by-side zh/en alignment, null bands behind every bar, and paper-grade export; (C) a **capture toolkit** (`--record` flag, a free-tier Colab notebook, community submissions) that outsources GPU cost; (D) **harness integration** repositioned from "must be live" to "record by default, live optional".

Milestones P1–P5 are ordered by citation leverage: P1 schema + record/replay (buildable now on mock data), P2 static gallery on Pages with deep links + export, P3 first real traces + Zenodo DOI (merged with the first M1 research numbers in one GPU rental), P4 Colab capture + community pipeline, P5 the full rigor layer. The platform's standing cost is zero; only capture costs money, and it is on-demand, batchable, and outsourceable. The platform amplifies visibility, but labs ultimately cite the **findings** (M1–M3) — which is why P3 fuses capturing citable traces with producing the first correlational results.
