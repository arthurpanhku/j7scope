# J7Scope 云 GPU 运行手册

> 更新时间：2026-07-27。价格与库存会变化，启动实例前以供应商控制台为准。

## 1. 结论与推荐配置

不需要购买实体 GPU。J7Scope 把昂贵计算集中在拟合和 trace 采集阶段，最适合租用
按秒或按小时计费的云 GPU，任务结束即释放。

| 任务 | 最低建议 | 稳妥选择 | 说明 |
|---|---:|---:|---|
| 上游 Qwen3.5-4B 已知案例复现 | 16 GB | 24 GB L4 / 3090 | 只加载已拟合 lens，先验证读出质量 |
| Qwen2.5-7B position-local 采集 | 24 GB | 40–48 GB | bf16 模型约 14 GB，反向与激活还需空间 |
| 7B paper estimator 正式拟合 | 40 GB | 48–80 GB | 可降低 `dim_batch` 换显存，但运行会更慢 |

当前首选是 **Runpod Pod + 48 GB A40/A6000**：交互式 shell、磁盘持久化和按小时
计费都适合一次性研究批次。2026-07-27 官方页面列出的 Pod 参考价为 A40
48 GB `$0.44/h`、RTX A6000 48 GB `$0.53/h`、L40S 48 GB `$0.99/h`；
24 GB L4 为 `$0.39/h`。这只是 GPU 实例标价，磁盘等费用另计。

官方链接：

- [Runpod GPU 定价](https://www.runpod.io/pricing)
- [Runpod Pod 管理文档](https://docs.runpod.io/pods/manage-pods)
- [Google Colab 资源限制说明](https://research.google.com/colaboratory/faq.html)

## 2. 直接租用：Runpod

这条路径无需提交研究资助申请，注册、充值后即可创建实例。

1. 在 [Runpod](https://www.runpod.io/) 注册账户并在 Billing 添加支付方式/余额。
2. 进入 **Pods → Deploy**，选择官方 PyTorch 模板。
3. 上游 4B 复现选择 24 GB；正式 7B 拟合优先选择 48 GB。磁盘建议至少 80 GB。
4. 启动后打开 Web Terminal 或 SSH，执行：

```bash
git clone https://github.com/arthurpanhku/j7scope.git
cd j7scope
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[upstream,fit]'

export HF_HOME=/workspace/hf_cache
python experiments/reproduce_upstream.py --preflight-only
python experiments/reproduce_upstream.py
```

成功后会生成 `results/upstream-reproduction.json`，其中记录实际 GPU、模型 commit、
lens snapshot、各层 J-lens/logit-lens top token 和 `euro` 概念命中情况。

## 3. 正式拟合与断点恢复

先用 48 GB 卡执行显存预检，再用两条语料测峰值显存和单条耗时。benchmark 使用独立
checkpoint，避免与正式 1000 条语料的 corpus SHA-1 冲突：

```bash
python experiments/fit_paper_jacobian.py --preflight-only

python experiments/fit_paper_jacobian.py \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --max-prompts 2 --min-chars 200 \
  --checkpoint results/jacobian-benchmark.checkpoint.pt \
  --output results/jacobian-benchmark.pt
```

`results/jacobian-benchmark.json` 会记录实际耗时和 `peak_cuda_memory_gb`。确认显存余量后
运行正式 1000 条拟合；若 Pod/SSH 中断，使用完全相同的命令即可从下一个 prompt
继续：

```bash
python experiments/fit_paper_jacobian.py \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --dataset-revision main \
  --max-prompts 1000 --min-chars 200
```

正式归档时应把 `--model-revision` 和 `--dataset-revision` 从 `main` 换成运行前确认的
commit。即使远端数据发生变化，checkpoint 的完整有序语料 SHA-1 也会拒绝混合续跑。
`dim_batch=8` 显存不足时可以降低到 4 或 2；这不改变估计定义，但会增加运行时间。

默认的 `data/jacobian_fit_smoke.jsonl` 只有 12 条中英语句，只用于验证代码、输出格式
和恢复机制，不足以形成研究结论。正式结果包括：

- `results/jacobian-qwen2.5-7b-l18.checkpoint.pt`：可恢复的累积和与进度；
- `results/jacobian-qwen2.5-7b-l18.pt`：sidecar 可直接加载的最终 float32 矩阵；
- `results/jacobian-qwen2.5-7b-l18.json`：revision、语料/矩阵 SHA-1、硬件和性能元数据。

加载正式矩阵：

```bash
cd apps/serve
python -m j7scope_serve --backend hf \
  --model Qwen/Qwen2.5-7B-Instruct --layer 18 \
  --jacobian-file ../../results/jacobian-qwen2.5-7b-l18.pt
```

开始大下载前先运行 `--preflight-only`；显存不足时脚本会直接退出。复制回结果和需要
保留的 cache 后，在控制台 **Terminate Pod**。只停止实例仍可能产生磁盘费用，操作前
查看控制台的实时费用摘要。

公共 Qwen 模型与公开 lens 不需要 Hugging Face token。若以后使用受限模型，把 token
放入平台 Secret/环境变量，禁止写入仓库、notebook 或 trace。

## 4. Colab 能否使用

可以，但建议只用于 P4 社区采集笔记本和小规模预演：

- 免费层 GPU 类型和配额不保证，昂贵资源受限；
- 免费 notebook 最长通常不超过 12 小时，实际还受可用性和使用模式影响；
- 即使是付费方案，具体 GPU 仍取决于当时库存。

如果 Colab 分配到 24 GB L4，可以运行上游 4B 复现；若只分配到约 16 GB T4，不建议
强行跑正式批次。P3 需要固定硬件、不中断和可核算成本，因此使用 Pod 更合适。

## 5. 申请免费的学术云额度

直接租用可以马上开始；学术额度适合覆盖后续 M1–M3 批量实验。

### Google Cloud Research Credits（优先）

1. 用学校/研究机构身份创建 Google Cloud 账户和 Billing Account。
2. 用官方 Pricing Calculator 估算 GPU、CPU、磁盘和存储费用。
3. 通过 [GCP Research Credits 申请指南](https://support.google.com/google-cloud-higher-ed/answer/10724468)
   的在线表单提交研究计划和费用估算。
4. 官方说明通常需要 6–8 周审核；申请全年开放。符合条件的博士生可申请每年
   `$1,000` GCP credits。

建议申请 `$1,000`，项目类型写成“finite proof-of-concept + repeatable open research
tool”，强调 trace、代码、实验配置和 Zenodo 数据都会公开。

### AWS Cloud Credit for Research

1. 先建立 AWS 账户并取得 12 位 account ID。
2. 使用学校/机构邮箱填写
   [AWS Cloud Credit for Research](https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research/)
   申请。
3. 研究生申请上限为 `$5,000`，教师/全职研究人员没有该上限。

AWS 官方当前说明无法处理 **greater China region** 的申请；如果申请人/机构地址属于
该范围，应优先走 GCP 或直接租用 Runpod，并在提交前向 AWS 确认香港地址是否可受理。

### NVIDIA Academic Grant

[NVIDIA Academic Grant Program](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/)
按主题和申请周期征集 proposal，适合中长期研究合作或硬件支持，不适合作为眼前 P3
批次的唯一依赖。

## 6. 申请材料清单

- 项目名称与 150–250 字英文摘要；
- 研究问题：跨语言 J-space 是否共享；
- 有限、可验证的交付物：上游复现、M1 指标、真实 trace、Zenodo DOI；
- 模型、GPU 类型、预计 GPU 小时和费用计算；
- 数据治理：只使用公开模型、公开/合成语料，不处理个人敏感数据；
- 开放科学计划：Apache-2.0 代码、版本化 schema、公开 artifact 和可复现实验配置；
- 时间线：复现 → 小批次 → 全概念 → DOI 发布。

可直接使用 [`gpu-credit-proposal.md`](gpu-credit-proposal.md) 作为英文申请底稿。
