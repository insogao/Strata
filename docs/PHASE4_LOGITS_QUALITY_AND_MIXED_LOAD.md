# P4：风险序数评测与单引擎混发开发验收

2026-10-16。分支 `codex/p4-ordinal-mixed-evaluation`；基于 PR #8
`18e1f88edb687d51a1f02b2d29e322f2b5947c00`，待独立审查。不合并 main。

**结论：数据已经找到，评测链路能运行；尚无可证明来自人工裁定的业务准确率。
混发的首个解释续写方案 NO-GO，不能把 EOS 空回答包装成吞吐提升。**

| 问题 | 本轮回答 |
|---|---|
| GEN-JSON 与 Logits 哪个更准、更快？ | 四条开发样本 Logits 观察耗时较低；没有人工金标、代表性或充分样本，不能裁决业务质量/稳定收益 |
| 同 Prompt GEN-LABEL 与 Logits？ | 实际输入 Token 哈希逐例相同，均返回3；生成约1.051s、Logits约0.882s，仅四条开发观察 |
| 人工金标的 Exact、±1、MAE、风险3漏报？ | **BLOCKED_NO_GOLD_DATA**；下表只对照合成设计标签 |
| M0→M1→M2？ | 请求解释的8个分支均只返回 EOS，修正后解释流程有效完成数为0，**NO-GO** |
| 是否做多序列/Chunked Prefill？ | 没有支持 M3 的证据；当前先解决数据/解释输出合同，不改原生调度器 |

## P4-1：实际数据盘点

用户指明 `D:/work/jev` 后，直接只读 SQLite，未调用其会修改状态的任务 CLI。
数据库 `crime_chat_dataset/registry/dataset_registry.sqlite` 的事务快照为：

| 状态 | 条数 |
|---|---:|
| accepted | 3387 |
| review | 66 |
| rejected | 107 |
| 合计 | 3560 |

119条 pairs。README/AGENTS 的“200条/6 review”是旧快照。
`generated/accepted/` 实有3453个 YAML，包含66条当前 DB=review；不能按目录名导入。
3387 accepted中，2条 YAML ParserError 原件保留；余3385条全部可读、风险标签与数据库一致、无规范化对话重复。
风险0/1/2/3分别为 **166 / 2208 / 543 / 468**。源类型为6条 synthetic_pilot、3379条 synthetic_subagent。

`labels.risk` 是案例设计标签；生成前指定，并有程序/模型判官过程。
**accepted 不等于人工终审，也不能证明双判官证据逐案全部闭合。**
`report/QUALITY_ISSUES.md` 还记有模板重复、算术错误、判官错号、争议记录。
旧 `data/benchmark.jsonl` 的194条作为另一套历史 benchmark 保留，未并入人工 C。
本轮没找到可绑定人工裁定记录的0–3金标，不把公开安全数据源的标签直接映射成本项目的序数金标。

导入只取 `conversation` 作为消息；labels/source/hidden_case 等不进入模型 Prompt。
保留源 YAML SHA、DB行快照 SHA、规则版本和来源类型。导入器保守标为 provisional / synthetic_design_label。
review/rejected/坏YAML均明确计入剔除原因，不改源库、不替用户裁决。

泄漏控制：规范化内容、conversation、已登记 pair、family/lineage/root/parent 连通分组。
3217条共用一组 family/lineage；再考虑119对 pairs，保守分组仅5组、最大3379条。
不能凭随机行切分声称独立测试集。`split` 可输出确定性分组清单，但不保证分层比例；
正式 final 入口本版本明确拒绝，待真实标注及血缘清理后独立注册一次正式活动。
本轮四条均为 **development**，未使用/消耗正式 holdout。

## 实现与复现

新增独立 opt-in 工具，不改 ScoreService、UnifiedService、原生内核、默认入口或 Profile：

- `tools/strata_risk_eval.py`：import-jev、audit、split、run。
- 后端同名模块：已有统一 HTTP API 的 A0/A1/B1 配对驾驶器，单次最多64个开发样本；超出须另登记活动。
- `strata_risk_data.py`：SQLite URI mode=ro/query_only/事务快照；YAML identity/risk/status 与 DB 对账；私有消息与公开统计分开。
- `strata_ordinal_metrics.py`：混淆矩阵、Exact、±1、MAE/MSE、严重偏差、风险3→2/→0/1、无效输出、聚类配对 bootstrap。
- `tools/strata_mixed_load.py`：客户端数/队列容量1..8、单个串行 workflow worker，整个 SCORE/Continue/release 不交叉；本地队列满显式503、无重试。
- `verify_risk_evidence.py`：公开脱敏逐例统计的独立算术/SHA核验，不宣称替代本地 native raw 复核。

```powershell
python -m pip install -r backends/qwen35_llama/tools/requirements-evaluation.txt
python tools/strata_risk_eval.py import-jev --root D:/work/jev --out <新的本地私有目录>
python tools/strata_risk_eval.py audit --data <该目录>/private-dataset.jsonl
python tools/strata_risk_eval.py split --data <该目录>/private-dataset.jsonl --out <新的私有split.json>
python tools/strata_risk_eval.py run --data <开发JSONL> --config <冻结模板JSON> --spec <已注册GPUspec> --out <新的私有目录> --enable-gpu --lifetime 300
python tools/strata_mixed_load.py --data <开发JSONL> --config <冻结模板JSON> --spec <已注册GPUspec> --out <另一新目录> --enable-gpu --lifetime 300 --clients 2 --capacity 2 --explain-max 64
python backends/qwen35_llama/tools/verify_risk_evidence.py backends/qwen35_llama/tests/evidence/risk_20261016
```

模板配置显式包含 a0_system、label_system、a0_max_new、label_max_new、explanation_suffix。
A0系统逐字保留历史业务 JSON 规则；A1/B1仅替换输出合同为“只回答一个数字”，使用 GGUF 自带 chat template、enable_thinking=false。
禁止自动安装 Profile、截断/填充输入、猜测候选 IDs、在生成后强制归一化数字。
`run` 必须显式启用GPU，并记录完整模型/引擎/pack索引、工具/模板/数据哈希、配置、硬件、软件、租约、窗口与资源限制。
不支持自动断点重跑或重试；新输出目录必须不存在，原始失败保留。

空输出/非法风险不丢分母：Exact/±1对全量计数；MAE/MSE无效项没有自然距离，故同时报
valid-only描述值和 invalid=3/9 的保守罚分，明确命名，避免冒充数学上的唯一MAE。
严重偏差与 invalid 联合率、risk3 invalid 单列。争议数据不进主金标表，允许集合单列。
配对 bootstrap 按 conversation_id 整簇重采样，低于30簇 INCONCLUSIVE；没有金标就不算金标差值。
条件概率保留 calibrated=false，不设置自动业务放行阈值。

## P4-2：双基线开发冒烟

RTX4060 Laptop 8188MiB，Windows，生产 GGUF
`66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`；
原生 exe `ea06d39411e160f5784f9c4ef160f5e9bcc1746fb92fd9b798677aabb0dd0619`。
3200槽/t8/u4096/c4096、uniform、MTP OFF、temperature0；三个既有生产 env不变。
canonical GPU lease：单 native，上限300s，累计窗口1200s，等待≤60s。
模型只解析头部/tokenizer，未重复全文件哈希或下载。

四条来自实际本地合成库 accepted 完整案例，每档一条、实际A0输入782/817/672/768 tokens。
按DB顺序选首条满足400..2200 tokens者，无截断、填充、看预测挑样本。
它们不是3896-token代表负载，不能把以下数字外推4K主负载。
A1/B1输入为749/784/639/735 tokens，逐例哈希相同。
真实Tokenizer：数字0/1/2/3分别15/16/17/18；空格前缀分别 `[220,15]`…，数字非special且单Token。
SCORE全词表logprob由现有 `/v1/decisions` 的 topk=0 candidate 路径返回，不重写softmax。

P固定3896/out1预热后，交错 A0/A1/B1、B1/A1/A0。生成保留实际EOS和数量。
计时包含本地编码、HTTP、引擎工作、解析/清理；启动/模型加载单列。

| 方法 | n | 平均完整E2E ms | 中位数ms | 实际平均输出Token（含EOS） | 有效格式完成数 |
|---|---:|---:|---:|---:|---:|
| A0 GEN-JSON | 4 | 1765.62 | 1455.97 | 16 | 4 |
| A1 GEN-LABEL | 4 | 1051.16 | 1052.48 | 2 | 4 |
| B1 LOGITS | 4 | 882.07 | 905.41 | 0 | 4 |

这是同进程预热后**小样本**观察，不叫稳态/可靠p95或质量合格吞吐。
完整HTTP非流响应未独立采集TTFT/ITL，Continue响应未暴露native decode_ms，保留null；
native原始DONE可查，不用HTTP平均耗时编造token级时钟。

| 与合成设计标签对照（非人工准确率） | Exact | ±1 | MAE | MSE | ≥2级错误 | risk3→2 | risk3→0/1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 1/4 | 2/4 | 1.25 | 2.25 | 2/4 | 0/1 | 0/1 |
| A1 | 1/4 | 2/4 | 1.50 | 3.50 | 2/4 | 0/1 | 0/1 |
| B1 | 1/4 | 2/4 | 1.50 | 3.50 | 2/4 | 0/1 | 0/1 |

A1与B1均输出3；一致不是准确。risk3仅一个样本，不能用其召回1/1推断漏报率可接受。
这也是为什么不以更低耗时宣布业务 PASS。金标质量表全部n=0，明确 BLOCKED_NO_GOLD_DATA。

## P4-3：混发失败及测量修正

第二个独立300s租约，同四案例，M0 JSON全生成，M1 predicted>=2才解释，M2同策略/数据，2客户端/容量2、GPU仍单worker。
所有四案例B1均为3，**自然分支为100%解释**，未为了凑50/80/95%而改风险标签；未跑人工Oracle。
初版在数字助手前缀直接追加12-token解释指令，SCORE→GEN cont的 `reused` 等于原Prompt长度，`prompt_read=12`，原前缀未重放。
但是8个解释分支均只生成 EOS（1Token），**没有解释内容**。
一次CPU模板检查还发现“重新渲染完整历史对话”不能假设与当前前缀逐ID相同，故没有直接替换原状态或盲目重跑。

| arm | 提交 | 最初仪表计数 | 修正后流程有效完成 | 平均实际输出Token | 总wall ms |
|---|---:|---:|---:|---:|---:|
| M0 | 4 | 4 | 4 | 16 | 6451.45 |
| M1 | 4 | 4 | **0** | 1（EOS） | 4691.28 |
| M2 | 4 | 4 | **0** | 1（EOS） | 4659.33 |

最初驾驶器只验证风险可解析，误把空解释计成功。原summary/raw不覆盖；本轮明确登记仪表缺陷，
新增EOS/control-only空解释拒绝及单测，公开 `mixed_corrected` 计数为0。
分类结果可以存在，但解释流程没完成，不能计作满足输出策略的审核完成率。
M0与解释路径的输出合同也未达相同质量，**不计算减少工作收益或调度收益**。
M2平均排队约863.69ms、最大1203.30ms；零实际HTTP拒绝/超时，CPU压力测试另覆盖本地队列满503。
没有原生Batch、Prefill/Decode重叠、GPU并行或调度提速证据。

## 资源、回归与证据

2个native均exit0、Job子树归零、fence未保留；0 runtime重跑、0 CUDA/NaN/OOM、0资源越界。
raw DONE分别44/48个内部操作，均在每项56操作登记上限内；两次不同任务窗口不借预算。
整卡100ms采样峰值7982.36/7897.05MiB，host RSS14896.38/14912.31MiB，可用RAM最低4731.27/4745.21MiB。
它们是整卡采样高水位，非精确allocator峰值。
P固定3896/out1首/末Prefill：9513.5→2836.4ms，3490.8→2876.3ms；
首轮冷启动效应明显，末请求接近既有本机量级，**不作为与历史冻结基线严格配对的回归通过**。
本轮未改native/CUDA或稳定服务源码，因此不额外重建或重做独立长文本恒等研究。

226 CPU tests通过（最终72.897s；此前225项72.937s也通过），46项公开脱敏算术/SHA检查通过。原有两个证据验证器继续保留。
公开包：`backends/qwen35_llama/tests/evidence/risk_20261016/`；无消息、原案例ID、输入Token列表。
原件仅本地：

- `D:/work/Strata魔改/results/unified-p4-20261016/`：导入审计、完整私有JSONL、选择/模板、CPU日志、修正说明。
- `D:/work/StrataUnified/experiments/p4-20261016/paired/` 与 `mixed/`：预登记/原HTTP结果、原native raw、租约、资源、完整SHA索引。
- `executed-risk-eval.py` 是按预登记SHA复原核对一致的执行时版本；最终代码另有CPU报告字段与空解释判定补强，不声称旧GPU运行已经测过新增判断。

## P4-4：裁决和下一步

P4-1工具与真实Tokenizer/API冒烟通过；人工金标与独立holdout仍阻塞。
P4-2四样本开发测时完成，正式准确率/可靠性能胜负 **BLOCKED_NO_GOLD_DATA / INCONCLUSIVE**。
P4-3当前直接追加解释方案 **NO-GO**；工具可测，不能说混合输出优化产品完成。
下一最高价值动作：对这批现有标注补独立人工裁定及可验证血缘，再冻结一份公平的生成/Logits输出合同；
解释路径另登记一个有界新聊天轮次续写设计，验证真实非空输出与状态，然后再交错M1/M2测量。
无需为这些阻塞启动多卡或改原生批处理，不改生产路由/审核标签，也不自动放行。
