<p align="center">
  <img src="assets/logo.svg" width="168" alt="TvinnHugr logo" />
</p>

<h1 align="center">TvinnHugr</h1>

<p align="center">
  <b>是一个工作空间，还是一对双生工作空间？</b><br>
  检验语言模型的 J-space / 全局工作空间在中英双语下是否共享同一套概念表示。
</p>

<p align="center">
  <i>Is there one workspace, or twin workspaces? Testing cross-lingual generalization of the J-space / global workspace in language models.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="license" />
  <img src="https://img.shields.io/badge/status-early%20scaffold-orange" alt="status" />
  <img src="https://img.shields.io/badge/model-Qwen2.5--7B--Instruct-green" alt="model" />
</p>

---

古北欧语 **_tvinnr_**（双生的、成对的、合股的线）+ **_hugr_**（心智、思维）。名字即研究问题：模型的"全局工作空间"在中英双语下运行时，是同一个 _hugr_，还是各自独立的一对 _tvinnir hugir_？

> **一句话定位：** 独立复现并扩展 Anthropic 的全局工作空间研究，检验模型内部的"思维空间"在中英双语下是否共享同一套概念表示 —— 一个大部分可解释性研究者做不了的双语视角实验。

**当前状态：早期脚手架。** M1（相关性分析）正在 Qwen2.5-7B-Instruct 上搭建，尚未产出任何结论性数据。

## 1. 研究背景

Anthropic 2026 年 7 月的论文 [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) 提出 **J-lens**：把中间层残差流通过输入-输出 Jacobian 的期望线性映射到最终层坐标，再用模型自己的 unembedding 解码，读出模型"倾向于说但还没说"的概念。论文发现这些可读出的方向构成一个稀疏子空间（**J-space**，约占激活方差的 6–10%），具备远超随机的广播式读写连接，功能上对应认知科学里的"全局工作空间"（global workspace）。

论文的语料全部是英文网页文本，J-lens 的拟合和验证都在单一语言内完成。这留下了一个论文没有回答的问题：

> 如果 workspace 是模型真正的"概念层"表示，它应该是语言无关的 —— 同一个概念不管用中文还是英文触发，都应该落在 J-space 里相近的方向上。**这个假设从未被检验过。**

官方配套代码 [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens)（Apache-2.0）明确标注为"参考实现，不维护、不接受贡献"，公开的开源模型 demo 也只是单语言可视化工具。**这个跨语言方向目前是空的。**

## 2. 研究问题与假设

**核心问题：** 同一概念（如"欺骗""操纵""让步"）在中文语境和英文语境下触发的 J-space 方向，是共享的还是各自独立的？

| | 假设 | 含义 |
|---|---|---|
| **H1** | **语言无关** —— 中英文触发高度重叠的方向 | 支持论文"这是模型真正思维层"的核心主张 |
| **H2** | **语言特异** —— 中英文触发基本不重叠的方向 | 说明当前的 J-space 更接近"表层语言习惯"而非语言无关的概念空间，是对论文核心主张的重要反例 |
| **H3** | **部分共享 / 分级** —— 抽象概念（情绪、意图类）共享程度高，具体实体概念（人名、专名）共享程度低 | 最可能的真实情况，也是最有信息量的结果 |

## 3. 方法

### 3.1 模型选择

需要中英文都强、架构是稠密 transformer（非 MoE，简化 Jacobian 计算）、且被 nnsight / TransformerLens 良好支持。

- **首选：** Qwen2.5-7B-Instruct（阿里，中英双语能力均衡，社区支持成熟）
- **跨模型复现：** Yi-1.5-9B、InternLM2.5-7B 作为第二个数据点 —— 同一结论如果只在一个模型上成立，说服力有限

### 3.2 平行语料构建

构造中英文平行的探针 prompt 集，覆盖论文中验证过的几类概念方向（欺骗/操纵类、评估意识类、多跳推理类），外加服务 M3 的抽象/具体对照组。每条 prompt **严格控制句法结构对齐，只替换语言**，避免"语言差异"和"表达方式差异"混淆。详见 [`data/`](data/) 与 [`data/concept_taxonomy.md`](data/concept_taxonomy.md)。

### 3.3 分析方法（建议按顺序推进，先便宜后昂贵）

- **M1 相关性分析（便宜，先做）。** 用同一个 J_l（模型的输入输出 Jacobian，与语言无关，只拟合一次）分别读出中文 prompt 和英文 prompt 在对应位置的 top-k token。用 **CKA 或 SVCCA** 衡量两组读出方向的表示相似度，配合 **top-k 重叠率**作为直观指标。
- **M2 因果验证（贵，M1 有信号后再做）。** Activation patching：把中文语境下"操纵"概念的残差流，patch 进英文语境的前向传播，看 J-lens 读出是否依然是"操纵"类概念。如果 patching 后读出**跟着"概念"走而不是跟着"patch 来源语言"走**，这是比相关性更强的因果证据。
- **M3 分级验证（可选，时间富余再做）。** 对比抽象概念 vs 具体实体概念的跨语言重叠率差异，验证 H3。

### 3.4 指标

- CKA / SVCCA 表示相似度（**跨语言 vs 同语言不同 prompt 的基线对比**）
- Top-k 读出 token 重叠率
- Patching 后"读出跟随概念"的比例（因果指标）

> ⚠️ **统计陷阱（已写入 [`metrics.py`](tvinnhugr/metrics.py)）：** 当样本数远小于维度（本项目正是 n≈30、d≈3584）时，两组独立随机矩阵的 CKA 基线可高达 ~0.7。**裸的跨语言 CKA 数字没有意义，必须和打乱配对的 null 基线一起报告。**

## 4. 仓库结构

```
tvinnhugr/
├── assets/
│   └── logo.svg               # 双股缠绕 = tvinnr（合股线），中心 = 共享 workspace 核
├── data/
│   ├── probe_prompts_zh.jsonl # 30 对严格句法对齐的中英平行语料
│   ├── probe_prompts_en.jsonl
│   └── concept_taxonomy.md    # 抽象/具体概念分类，服务 M3
├── tvinnhugr/
│   ├── data.py                # 语料加载与配对
│   ├── artifacts.py           # 实验 run 的 JSON/JSONL 导出契约
│   ├── fitting.py             # J-lens 拟合，适配 Qwen2.5 等稠密模型
│   ├── patching.py            # M2 的跨语言 activation patching
│   ├── metrics.py             # CKA / SVCCA / overlap 计算
│   └── viz.py                 # 双语对照的读出可视化页面
├── experiments/
│   └── build_demo_run.py      # 生成前端开发用 demo artifact（非真实实验）
├── apps/web/                  # React/Vite J-Space Explorer
├── notebooks/
│   └── walkthrough_zh_en.ipynb
└── results/                   # 跑出来的图表和数据
```

## 5. 快速开始

```bash
pip install -e .
# 建议使用约 20 GB 显存的 GPU 以 bf16 加载 Qwen2.5-7B-Instruct
jupyter lab notebooks/walkthrough_zh_en.ipynb
```

```python
from tvinnhugr.data import load_parallel_pairs
from tvinnhugr.fitting import load_model, JLens

model, tok = load_model("Qwen/Qwen2.5-7B-Instruct")
jlens = JLens(model, tok, layer=18)
jlens.estimate_jacobian(corpus_prompts)          # 只拟合一次，语言无关

pairs = load_parallel_pairs("data")
h_zh = jlens.collect_residual(pairs["deception-01"]["zh"]["text"])
h_en = jlens.collect_residual(pairs["deception-01"]["en"]["text"])
print(jlens.readout(h_zh), jlens.readout(h_en))  # 两种语言的读出一致吗？
```

### 5.1 J-Space Explorer 前端

前端读取 `results/runs/<run_id>/` 下的一组稳定 artifact：

- `manifest.json`：模型、层、语言、类别、artifact 版本
- `readouts.jsonl`：每个 zh/en probe 的原始 J-lens top-k
- `patches.jsonl`：M2 activation patching 的概念迁移、语言保持、对照组结果
- `projections.json`：Python 侧预计算的 2D/3D J-space 坐标
- `layer_scan.json`：不同 patch layer 的成功率曲线
- `metrics.json`：run 级摘要指标

本地开发可以先生成一个明确标记为 demo 的假数据 run（不代表实验结论）：

```bash
python3 experiments/build_demo_run.py
cd apps/web
npm install
npm run dev -- --port 5173
```

打开 `http://127.0.0.1:5173/` 后可以查看：

- **Patching Matrix**：跨语言概念迁移与 random/unrelated 对照
- **Pair Explorer**：单个 prompt pair 的原始 readout、patched readout、next-token sanity channel
- **J-Space Map**：中英 readout/projection 的二维分布
- **Layer Scan**：共享信号随 patch layer 的变化

## 6. 路线图

- [ ] M1 跑通 Qwen2.5-7B-Instruct —— 先覆盖"欺骗/操纵"这一类概念
- [ ] M1 覆盖全部概念类别，带同语言基线对照
- [ ] M2 activation patching（M1 有信号后启动）
- [ ] M3 抽象 vs 具体的分级验证
- [ ] 跨模型复现（Yi-1.5-9B / InternLM2.5-7B）

## 7. 许可与归属

Apache-2.0（见 [LICENSE](LICENSE)）。fitting 逻辑基于 [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens)（Apache-2.0）的方法做二次开发，上游归属声明保留在 [NOTICE](NOTICE) 中。

> **TvinnHugr 是一个独立的第三方研究扩展，不是 Anthropic 官方项目，与 Anthropic 无隶属或背书关系。**
>
> **TvinnHugr is an independent third-party research extension. It is not an official Anthropic project, and is not affiliated with or endorsed by Anthropic.**

---

## English summary

**TvinnHugr** (Old Norse *tvinnr* "twined / paired" + *hugr* "mind") asks whether the **J-space / global workspace** described in Anthropic's July 2026 paper [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) is **shared across Chinese and English**, or maintained separately per language. The paper fits and validates its **J-lens** entirely on English text; whether the "concept level" it reveals is language-independent has never been tested. This is a Chinese-focused research extension, so the primary documentation above is in Chinese.

Three mutually exclusive hypotheses: **H1** language-independent (shared directions — supports the paper's core claim), **H2** language-specific (disjoint directions — a substantive counterexample), **H3** graded (abstract concepts shared, concrete entities language-bound — the most likely outcome).

Method, cheap-to-expensive: **M1** correlational (one language-agnostic Jacobian `J_l`, then CKA / SVCCA / top-k overlap between zh and en readouts, always reported against a shuffled-pairing null), **M2** causal (cross-lingual activation patching — does the readout follow the *concept* or the *source language*?), **M3** the abstract-vs-concrete gradient. Primary model Qwen2.5-7B-Instruct; replication on Yi-1.5-9B / InternLM2.5-7B. See the quickstart above and [`notebooks/walkthrough_zh_en.ipynb`](notebooks/walkthrough_zh_en.ipynb).
