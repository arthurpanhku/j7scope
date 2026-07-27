# Contributing to J7Scope

感谢你帮助扩展跨语言 J-space trace。最有价值的贡献是：可复现、明确标注局限、不会把
preview 当作研究结论的真实模型采集。

## 快速路径：Colab 采集

1. 在 Google Colab 打开 `notebooks/capture_colab.ipynb`。
2. 选择 **Runtime → Change runtime type → GPU**。
3. 按顺序运行所有 cell，填写 prompt、语言、概念和唯一 `trace_id`。
4. 下载 notebook 生成的 ZIP，在本地解压。

默认 Qwen2.5-1.5B + 少量随机探针用于验证贡献流程，产物会标记
`preview: true`，不是可引用的 M1 证据。GPU 型号、dtype、模型 revision、Jacobian
estimator 和张量 SHA-1 会写进 manifest。

也可以在任何 CUDA 机器直接运行：

```bash
pip install -e .
python experiments/capture_trace.py \
  --trace-id community-deception-en \
  --language en \
  --concept deception \
  --prompt "In one sentence, explain why deception can be tempting."
```

先用 `--dry-run` 检查配置，不会下载模型或占用 GPU。

## Trace 命名与内容

- `trace_id`：1–80 个小写字母、数字、点、下划线或连字符；建议
  `community-<concept>-<lang>-<short-id>`。
- prompt 不得包含个人信息、机密内容、API key 或未获许可的数据。
- 模型必须是贡献者有权使用和重新发布派生小型 artifact 的权重。
- 不提交模型权重、Jacobian `.pt`、Hugging Face cache 或其他大文件。
- 不手工修改 `tokens.jsonl` 中的 `rigor.sharedness`。严谨层只能由
  `j7scope.rigor` 生成。
- 社区 trace 默认保留 `preview: true`；研究级标记由维护者在复核协议、语料和
  estimator 收敛后处理。

## 加入 Gallery

把完整 trace 目录复制到 `results/traces/<trace_id>/`，然后从仓库根目录运行：

```bash
python -c "from j7scope.trace import rebuild_trace_index; rebuild_trace_index('results/traces')"
python experiments/validate_trace_gallery.py
pytest -q
```

提交 `manifest.json`、`tokens.jsonl`、`metrics.json`，以及平行 trace 才需要的
`align.json`。同时提交重建后的 `results/traces/index.json`。

## Pull request 检查清单

- [ ] Trace 在本地通过 `experiments/validate_trace_gallery.py`
- [ ] `trace_id` 与目录名一致且 index 已重建
- [ ] `is_demo` / `preview` 标记真实准确
- [ ] manifest 含 model revision、dtype、device、estimator 和 Jacobian SHA-1
- [ ] prompt 和 artifact 不含个人、机密或受限信息
- [ ] 未提交模型权重、cache 或大型二进制
- [ ] PR 描述说明模型、GPU、语言、概念和采集目的

## Code contributions

Python 变更应补测试；前端不得重算 sharedness/null 指标，只显示 trace 中已烤入的值。
运行：

```bash
pytest -q
cd apps/web && npm ci && npm audit --audit-level=high && npm run build
```
