# 心情实时看板

页面位于 `apps/site/mood.html`，与 Live、Connect 和 Gallery 共用现有服务，随 wheel 和静态站一起打包。
无需安装新的前端依赖。Gallery 和 Live 导航均有“心情看板”入口。

在仓库根目录启动：

```bash
PYTHONPATH=apps/serve .venv/bin/python -m j7scope_serve --backend mock --port 8799
```

打开 `http://127.0.0.1:8799/mood.html`。

页面有四个来源，切换时会清空趋势，避免混合不同来源的记录：

- **文字输入**：边输入边分析中英情绪关键词，包含简单否定处理。只在浏览器内运行，不上传、不持久化。
- **实时模型**：通过同源 `jspace/stream` 订阅当前会话。可分析最近 500 字符的输出文字，
  或当前 J-lens top-k 词汇读出；二者独立选择。可从页面发送一条文本消息触发模型生成。
  认证沿用 Connect 的 HttpOnly cookie，不在 URL 或浏览器存储中保存 key。
- **体验演示**：自动轮播合成的正面、负面、混合心情例句，支持暂停/继续。始终标注为 DEMO。
- **工具集成**：显示同一专用会话中最近一次来自 Codex、Claude Code 或其他工具的文字更新。

也可以直接使用 `?mode=live` 或 `?mode=demo` 链接。
完整看板和集成小窗右上角都有中英文切换。可以用 `?lang=en` 直接打开英文看板，
或用 `?mode=tools&view=compact&lang=en` 打开英文小窗。切换不重载页面，也不清空输入或趋势；
浏览器只保存语言偏好。工具发送的文字和关键词保留原文。
纯静态站可以使用文字输入和演示，实时模型需要运行 J7Scope 服务。
mock 后端提供合成读出，页面会显示“DEMO 合成”；真实模型沿用现有 HF backend 配置。

## 把小窗放在 Codex 或 Claude Code 旁边

点击看板右上角的 **集成小窗 ↗**，或打开 `http://127.0.0.1:8799/mood.html?mode=tools&view=compact`。
在支持弹窗的浏览器中，点小窗右上角的 **另开窗口 ↗** 可将它独立打开。也可直接在支持本地网页的编辑器面板里打开小窗地址。
小窗与工具并排工作，不修改工具的模型设置，也不需要把模型流量代理到 J7Scope。

### 1. 创建专用会话

在仓库根目录用同一终端启动认证服务并创建会话：

```bash
export J7SCOPE_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
PYTHONPATH=apps/serve python3 -m j7scope_serve --backend mock --port 8799 &
j7scope_pid=$!
# 等服务打印启动地址后，执行下面两行。
export J7SCOPE_MOOD_SESSION_KEY="$(printf 'Authorization: Bearer %s\n' "$J7SCOPE_API_KEY" |
  curl -fsS -X POST -H @- http://127.0.0.1:8799/jspace/sessions |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["api_key"])')"
unset J7SCOPE_API_KEY
```

打开同一服务的 [Connect 页面](http://127.0.0.1:8799/connect.html)，输入 `J7SCOPE_MOOD_SESSION_KEY` 的值，
点 **Connect** 后打开集成小窗。macOS 可用 `printf '%s' "$J7SCOPE_MOOD_SESSION_KEY" | pbcopy` 复制密钥。
重启服务后需重新创建会话。使用其他端口时，在工具进程的环境中设置 `J7SCOPE_MOOD_PORT`。
结束本地服务时可执行 `kill "$j7scope_pid"`。

先在同一终端发一个示例，小窗应立即显示来源和情绪：

```bash
printf '%s' '今天很顺利，我很开心！' |
  python3 apps/serve/j7scope_serve/mood_bridge.py text --source "Local demo"
```

### 2. 安装工具钩子

在目标项目目录运行已安装的自动接入命令（macOS / Linux）：

```bash
j7scope-integrate                                # 自动发现，预览文件路径
j7scope-integrate --apply                        # 写入项目级配置
j7scope-integrate --agent codex --agent claude --apply
j7scope-integrate --remove --apply                # 移除未修改的接入项
```

从源码也可以运行，不必安装 CLI。在仓库根目录执行：

```bash
PYTHONPATH=apps/serve python3 -m j7scope_serve.mood_integrate --project /path/to/project --apply
```

接入器从 PATH 和项目配置标记发现支持的 Agent，从当前路径向上查找 Git 根目录。
`--agent` 可覆盖自动发现，并可重复指定。它合并现有 JSON 配置，不覆盖自定义插件；
不支持的 JSON 或符号链接会原样保留并报错。不会修改全局配置、模型提供方或写入密钥。
OpenCode 适配器支持 V1 插件接口；如果项目使用其他版本，可先通过 `--agent` 只选择兼容的工具。

| 工具 | 配置位置 | 示例 |
|---|---|---|
| Codex | 将 `hooks.Stop` 合并到项目的 `.codex/hooks.json`；在 `/hooks` 里审阅并信任配置 | [Codex Stop 模板](../apps/serve/integrations/mood/codex-hooks.json) |
| Claude Code | 将 `hooks.Stop` 合并到项目的 `.claude/settings.local.json`，可通过 `/hooks` 检查 | [Claude Code Stop 模板](../apps/serve/integrations/mood/claude-settings.json) |
| OpenCode V1 | 在 `.opencode/plugins/` 添加项目插件，重启 OpenCode 后加载 | [OpenCode V1 插件](../apps/serve/j7scope_serve/mood_opencode.mjs) |

把模板中的路径替换成真实绝对路径，保留已有钩子。让运行钩子的进程继承 `J7SCOPE_MOOD_SESSION_KEY`；
终端版本可从上面设置环境变量的终端启动。图形应用需要其本地钩子运行环境也能读取该变量。
不要把密钥写进仓库或钩子 JSON。

Codex 和 Claude Code 通过 `Stop` 的 `last_assistant_message` 更新。
OpenCode V1 在 `session.idle` 时用 SDK 读取该轮对应的助手消息，过滤工具、推理和合成片段。
这些适配器只向 `127.0.0.1` 发送最后 500 个字符，不读历史文件，不转发提示词或工具输入。
回复片段会在该会话的内存事件队列中短暂保留；不会写入 trace 文件。连接失败时钩子不阻塞工具回复。
参见 [OpenAI Docs：Codex Hooks](https://learn.chatgpt.com/docs/hooks#stop)、
[Claude Code Hooks](https://code.claude.com/docs/en/hooks#stop) 和 [OpenCode V1 Plugins](https://opencode.ai/v1/docs/plugins/)。

### 3. 接入其他工具

可用上面的 `text` 命令，或向本机服务发送 `POST /mood/events`：

```json
{"source":"My tool","text":"最新的回复片段","phase":"update"}
```

使用 `Authorization: Bearer <专用会话密钥>`，成功返回 HTTP 202。文字最多 500 个字符。
`phase` 默认为 `complete`；支持增量输出的工具可在生成中反复推送当前文字片段，标为 `update`。
窗口会立即显示每次推送；内置的 Agent 适配器在回合结束时更新。
外部工具模式基于输出文字，只有直接通过 J7Scope 模型服务生成时才有逐 token 的 J-lens 读出。

### 4. 在 Codex 回复过程中更新

用 `j7scope-codex-live` 开一个独立的 Codex App Server 对话。打开小窗的 **Agent tools**，
在运行命令的终端输入消息。小窗随助手回复的文字片段更新，回合结束时显示完成状态。
普通 Codex 桌面聊天仍使用已配置的回合结束钩子。

```bash
j7scope-codex-live --session-key-file "/path/to/session-key"
```

也可通过 `J7SCOPE_MOOD_SESSION_KEY` 提供专用密钥，从源码运行时用
`PYTHONPATH=apps/serve python3 -m j7scope_serve.codex_live`。
如本机 Codex 默认模型不可用，用 `--model` 为这次对话选择可用模型。
流式入口使用 Codex 的只读沙盒，支持连续多轮，空行退出。

服务启动时加 `--mood-embeddings` 可在 **Vector-space words** 区域显示近邻词。
此选项需要安装 HF 依赖，并在本机缓存 `sentence-transformers/all-MiniLM-L6-v2`。
服务用 MiniLM 编码最近 500 个字符，再按余弦相似度从固定英文词表选出六个近邻。
此分数来自独立的文字编码器。托管 Codex 的流式接口提供回复文字，本地 HF 模型则可以在 **Live model** 中显示 J-lens top-k。
见 [Codex App Server 流式事件](https://learn.chatgpt.com/docs/app-server#streaming-notifications)。

## 显示与判断规则

表情和配色分别表示“好心情”“坏心情”“心情有点复杂”“等待信号”。
情绪指数在 −100 到 +100 之间，是 `(正面权重 − 负面权重) / 总权重 × 100`，
不是概率、置信度或临床量表。文字每次关键词命中权重为 1；J-lens 模式仅匹配完整情绪词，
权重为 `1 / (global_rank + 1)`，不把 logit 数值当情绪概率。
指数大于 20 为正面，小于 −20 为负面，其余为混合；无匹配时不显示数值，也不强判好坏。
页面同时展示命中的词和最近 32 次更新趋势。

断线会撤销当前判断；生成过程中 15 秒无新 token 会显示“信号已暂停”。
新一轮生成清空上一轮观察文字；完成后明确显示“本轮结果”，中断时撤销判断。
重复 token 事件按 request ID / seq 去重，切换来源后会关闭旧连接、停止演示并取消页面等待中的请求。
取消浏览器请求不保证后台推理立刻停止。

SSE token 事件新增兼容字段 `text`，保留准确的表面文本（包括空格），原有 `token` 字段不变。
旧服务没有 `text` 字段时会回退到拼接 `token`，其空格还原效果取决于后端。

这是可解释的词汇规则原型，没有训练或验证过情绪分类器。
反讽、引用、复杂否定、主语和语境可能造成误判；J-lens 的孤立词也不包含完整否定语境。
结果描述输入文字或词汇读出中的情绪倾向，不能证明人或模型的真实心理状态。

## 检查

```bash
node --test tests/test_mood.cjs
npm test
```

Node 测试覆盖中英正负情绪、否定、混合/无信号、精确 token 匹配、rank 加权、非有限输入与指数边界。
HTTP 测试检查 SSE 文本与真实 chat response 的空格及内容一致。
