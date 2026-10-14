# Qwen35-llama 后端导入冒烟＋导入保真度对照（task 355，2026-10-14）

**一行状态**：native 3/3 全部 exit 0（A 旧 exe `03bd7068` ×2 次＝1 次采样器故障+1 次完整；B 新 exe `6062a714` ×1），无 fence/无残留；两例可见 token id 序列、文本、finish、hits/misses、cache、peak VRAM 全部一致；计时存在小样本偏差（req1 TPS B 偏低，见 §4 c）。

## 1. 设计与执行事实

- A＝旧实验 exe `results/phase2_mtp/consistentSlice300/strata.exe`（sha256 `03bd7068…`，139,107,840 B）；B＝354 新仓构建 `D:/work/StrataUnified/strata/backends/qwen35_llama/build/strata.exe`（sha256 `6062a714…`，139,208,192 B）。
- 两例 argv/env/cwd 逐项一致，唯一差异＝exe 路径（`compare-AB.json`：`argv_equal(tail)=True`、`argv0_differs=True`、`env_equal=True`、`env_diff={}`）；argv/env 快照：`pre-env/pre-A.json`、`pre-env/pre-B.json`。
- 12 项 env gate 全开列表与 347/348 一致，REJECTED 门 `STRATA_QWEN35_SLICE_GU_MULTI=0` 保持关；清 `STRATA_*`/`GGML_*`/`LLAMA_GRAPH*`/`CUDA_LAUNCH_BLOCKING` 后逐项设置；PATH 前置 `/d/work/Strata魔改/runtime/cuda-12.8.1/bin`；child 去 `STRATA_OPENCODE_OBSERVER_LOG_DIR`。
- 单 GPU owner：`tools/gpu-lock.py`（sha256 `337426d6…`）`run` 全流程，`max_launch=1`、`runtime_limit_seconds=300`、wait≤60s；每 native 一租约（wait attempts 1, wait 0.0 s），child cleanup `all_exited=true` / `active_processes_zero=true`，marker 均 auto-removed（`retained=false`）。
- RAM 门（pre-run AvailPhys ≥8 GiB）：A=18179.9 MB（True），B=18195.8 MB（True）；10 s 采样 CSV 与 postfallback 见 §3.3 与 `samples/`、`batch/`。
- 窗口：task.md mtime 2026-10-14T16:24:30.226082+08:00 起，批末 elapsed 9.08 min（<20，无超窗）；批末 lock 1 B 空、无 marker、无 strata 进程、GPU 1127 MiB util 0pct。
- **A 执行两次的登记**：A 第 1 次（attempt1）因我方采样器 `Out-File -Path` 参数名 bug 未产生运行中 CSV 采样（native 本体 exit 0、引擎数据完整），修复后按同 argv/env 重跑 A 一次作为正式 A；attempt1 原件保留在 `attempt1-A/`（child_00.log/result-A.json/ready/采样摘要等），≤4 native 上限内用了 3 次。
- 未改动任何源码（新仓 backends/**、研究仓源码、官方文件均只读）；新仓仅新增本报告 `docs/backends/qwen35_llama_smoke.md`。

## 2. 两例原文行摘录（verbatim，均出自 `raw/raw-*/child_00.log`）

### 2.1 A（旧 exe 03bd7068）

READY 类启动行：
```
strata qwen35: bridge ready: 40 layers x 256 experts, 2048 x 512 experts top-8, workers 7, native tier on, prompt experts GPU/MMQ (grouped, resident cache + bounded cold ring)
```
prefill 段起始：
```
{"event":"prefill_start","since_process_start_seconds":4.6441715}
```
`qwen35_mtp_txn_request_timing`（照录）：
```
{"event":"qwen35_mtp_txn_request_timing","request":1,"anchor":"prefill_begin","reset_seconds":-1,"prefill_process_seconds":3.7672537,"prefill_finish_seconds":0.0822442,"prefill_seconds":3.8526796,"seed_sample_seconds":3.85414,"engine_entry_seconds":3.8546342,"ttft_flush_request_seconds":3.8609597,"eos_complete_request_seconds":4.2857425,"request_end_seconds":4.2906051,"json_closed":true,"json_close_flush_request_seconds":4.2855128,"engine_wall_seconds":0.4311083,"rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","eog_id":248046}
{"event":"qwen35_mtp_txn_request_timing","request":2,"anchor":"request_reset","reset_seconds":0.0015496,"prefill_process_seconds":2.7496548,"prefill_finish_seconds":0.0833571,"prefill_seconds":2.8358097,"seed_sample_seconds":2.8368602,"engine_entry_seconds":2.8373588,"ttft_flush_request_seconds":2.8424485,"eos_complete_request_seconds":3.2905886,"request_end_seconds":3.2966132,"json_closed":true,"json_close_flush_request_seconds":3.2902939,"engine_wall_seconds":0.4532298,"rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","eog_id":248046}
```
`qwen35_mtp_txn_request_flush_arrays`（照录，逐 token flush 时间数组）：
```
{"event":"qwen35_mtp_txn_request_flush_arrays","request":1,"engine_entry_seconds":3.8546342,"emit_ready_engine_seconds":[0.005661,0.0789863,0.0805372,0.1514917,0.1526996,0.2014627,0.2023885,0.2505508,0.2516446,0.2949304,0.2958943,0.3415364,0.3426078,0.3871303,0.3882285,0.43018],"emit_flush_engine_seconds":[0.0063255,0.0796707,0.0812705,0.1522684,0.1534739,0.2021715,0.2030841,0.251233,0.2524443,0.2956271,0.2966562,0.3423293,0.3433761,0.3879387,0.3890267,0.4308786],"emit_ready_request_seconds":[3.8602952,3.9336205,3.9351714,4.0061259,4.0073338,4.0560969,4.0570227,4.105185,4.1062788,4.1495646,4.1505285,4.1961706,4.197242,4.2417645,4.2428627,4.2848142],"emit_flush_request_seconds":[3.8609597,3.9343049,3.9359047,4.0069026,4.0081081,4.0568057,4.0577183,4.1058672,4.1070785,4.1502613,4.1512904,4.1969635,4.1980103,4.2425729,4.2436609,4.2855128],"cost_split":false,"cost_sync":false,"cost_phases_ms":{"initdraftctx":0,"draft":0,"verify":0,"hcopy":0,"replay":0,"rollback":0,"emit":0,"unattributed":0},"cost_sync_calls":0,"cost_sync_wait_ms":0}
{"event":"qwen35_mtp_txn_request_flush_arrays","request":2,"engine_entry_seconds":2.8373588,"emit_ready_engine_seconds":[0.0044281,0.0585533,0.0597886,0.109687,0.1106443,0.1549778,0.1559254,0.2041023,0.2052088,0.2474992,0.2484645,0.3371117,0.3394411,0.3998785,0.4017184,0.4518451],"emit_flush_engine_seconds":[0.0050897,0.0592322,0.0604861,0.1104179,0.1113732,0.1556706,0.156637,0.2047858,0.2060199,0.2482197,0.2492223,0.3389555,0.3410602,0.4013081,0.4030709,0.4529351],"emit_ready_request_seconds":[2.8417869,2.8959121,2.8971474,2.9470458,2.9480031,2.9923366,2.9932842,3.0414611,3.0425676,3.084858,3.0858233,3.1744705,3.1767999,3.2372373,3.2390772,3.2892039],"emit_flush_request_seconds":[2.8424485,2.896591,2.8978449,2.9477767,2.948732,2.9930294,2.9939958,3.0421446,3.0433787,3.0855785,3.0865811,3.1763143,3.178419,3.2386669,3.2404297,3.2902939],"cost_split":false,"cost_sync":false,"cost_phases_ms":{"initdraftctx":0,"draft":0,"verify":0,"hcopy":0,"replay":0,"rollback":0,"emit":0,"unattributed":0},"cost_sync_calls":0,"cost_sync_wait_ms":0}
```
`qwen35_mtp_txn_done`（照录）：
```
{"event":"qwen35_mtp_txn_done","rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","end_pos":3912,"max_new":256,"wall_s":0.4311083}
{"event":"qwen35_mtp_txn_done","rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","end_pos":3912,"max_new":256,"wall_s":0.4532298}
```
`summary`（照录）：
```
{"event":"summary","exit":0,"prompt_tokens":3896,"output_tokens":32,"decode_evaluations":0,"prefill_seconds":3.8526796,"decode_seconds":3.7579166,"ttft_seconds":3.8609597,"prefill_tokens_per_s":1011.24,"decode_evaluations_per_s":0,"strata_jobs":2493440,"strata_seconds":6.22271,"decode_layers":0,"decode_hits":3654,"decode_misses":6586,"prefill_layers":80,"prefill_entries":2493440,"cache_slots":2700,"cache_bytes":3354632192,"cache_mib":3199.2,"cache_payload_mib":3199.2,"cache_resident":2700,"hit_overlap":0,"hit_late":0,"hit_inflight":0,"all_hit_layers":0,"prefill_gpu_layers":80,"prefill_resident":4292,"prefill_streamed":14698,"prefill_product_calls":2460,"prefill_stream_mib":17626.91,"prefill_workspace_mib":539.37,"prefill_ring_mib":53.75,"prefill_loan_slots":479,"prefill_loan_mib":566.61,"prefill_owned_mib":0.00,"prefill_loan_restored":479,"prefill_loan_refill_ms":76.2,"prefill_loan_why":"","tier_ms":{"ids":160.7,"launch":51.5,"misses":320.3,"combine":11.0,"admit":0.0},"vram_samples":4,"peak_device_vram_mib":7577.3,"device_total_mib":8187.5,"prefill_process_seconds":3.7672537,"prefill_process_calls":1,"prefill_finish_seconds":0.0822442,"decode_eval_seconds":0,"decode_eval_spans":0,"decode_eval_per_s":0,"decode_observe_seconds":0,"decode_observe_rounds":0,"decode_first_vram_seconds":2.4e-06,"decode_first_logits_wait_seconds":7.9e-06,"decode_vram_seconds":2.4e-06,"decode_vram_samples":1,"vram_sample_seconds_total":0.0001697,"engine_perf_t_eval_ms_first":0,"engine_perf_t_eval_ms_last":0,"engine_perf_t_eval_ms_delta":0,"engine_perf_n_eval_first":1,"engine_perf_n_eval_last":1,"engine_perf_n_eval_delta":0,"engine_perf_t_p_eval_ms_first":0.028,"engine_perf_n_p_eval_first":3896,"shared_overlap_marker_layers":0,"shared_overlap_join_layers":0,"decode_scan_seconds":0.0004213,"decode_scan_rounds":1,"decode_sampler_seconds":0.0009246,"decode_sampler_rounds":1,"decode_topk_seconds":0,"decode_topk_rounds":0,"decode_piece_json_flush_seconds":0,"decode_piece_json_flush_rounds":0,"decode_piece_seconds":0,"decode_piece_rounds":0,"decode_piece_hex_seconds":0,"decode_piece_hex_rounds":0,"decode_token_write_seconds":0,"decode_token_write_rounds":0,"decode_flush_seconds":0,"decode_flush_rounds":0,"mtp_txn":true,"legacy_aggregate_authoritative":false}
```

### 2.2 B（新 exe 6062a714）

READY 类启动行：
```
strata qwen35: bridge ready: 40 layers x 256 experts, 2048 x 512 experts top-8, workers 7, native tier on, prompt experts GPU/MMQ (grouped, resident cache + bounded cold ring)
```
prefill 段起始：
```
{"event":"prefill_start","since_process_start_seconds":4.2235177}
```
`qwen35_mtp_txn_request_timing`（照录）：
```
{"event":"qwen35_mtp_txn_request_timing","request":1,"anchor":"prefill_begin","reset_seconds":-1,"prefill_process_seconds":4.0038373,"prefill_finish_seconds":0.0823802,"prefill_seconds":4.0881237,"seed_sample_seconds":4.0896164,"engine_entry_seconds":4.0901187,"ttft_flush_request_seconds":4.0960101,"eos_complete_request_seconds":4.6883087,"request_end_seconds":4.6935349,"json_closed":true,"json_close_flush_request_seconds":4.6877699,"engine_wall_seconds":0.59819,"rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","eog_id":248046}
{"event":"qwen35_mtp_txn_request_timing","request":2,"anchor":"request_reset","reset_seconds":0.0016224,"prefill_process_seconds":2.7882773,"prefill_finish_seconds":0.0847682,"prefill_seconds":2.8761734,"seed_sample_seconds":2.877206,"engine_entry_seconds":2.8778123,"ttft_flush_request_seconds":2.8845427,"eos_complete_request_seconds":3.3470058,"request_end_seconds":3.3521339,"json_closed":true,"json_close_flush_request_seconds":3.3466904,"engine_wall_seconds":0.4691935,"rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","eog_id":248046}
```
`qwen35_mtp_txn_request_flush_arrays`（照录，逐 token flush 时间数组）：
```
{"event":"qwen35_mtp_txn_request_flush_arrays","request":1,"engine_entry_seconds":4.0901187,"emit_ready_engine_seconds":[0.0052185,0.0681612,0.0696352,0.1354263,0.1364766,0.2581029,0.2603651,0.3354528,0.3376601,0.4088411,0.4106644,0.4674642,0.4693989,0.5365939,0.5384128,0.5964889],"emit_flush_engine_seconds":[0.0058914,0.0688385,0.0703683,0.1362385,0.1372146,0.2599126,0.2621375,0.3370496,0.3392925,0.4102616,0.4122001,0.4690643,0.4709476,0.5378383,0.5396264,0.5976512],"emit_ready_request_seconds":[4.0953372,4.1582799,4.1597539,4.225545,4.2265953,4.3482216,4.3504838,4.4255715,4.4277788,4.4989598,4.5007831,4.5575829,4.5595176,4.6267126,4.6285315,4.6866076],"emit_flush_request_seconds":[4.0960101,4.1589572,4.160487,4.2263572,4.2273333,4.3500313,4.3522562,4.4271683,4.4294112,4.5003803,4.5023188,4.559183,4.5610663,4.627957,4.6297451,4.6877699],"cost_split":false,"cost_sync":false,"cost_phases_ms":{"initdraftctx":0,"draft":0,"verify":0,"hcopy":0,"replay":0,"rollback":0,"emit":0,"unattributed":0},"cost_sync_calls":0,"cost_sync_wait_ms":0}
{"event":"qwen35_mtp_txn_request_flush_arrays","request":2,"engine_entry_seconds":2.8778123,"emit_ready_engine_seconds":[0.0059271,0.0849951,0.086669,0.1504565,0.1519415,0.2104246,0.2119217,0.268254,0.2696497,0.3195724,0.3208991,0.3702561,0.3717737,0.4200042,0.4215229,0.4678248],"emit_flush_engine_seconds":[0.0067304,0.0858413,0.0875675,0.1514481,0.1529376,0.2114249,0.2129873,0.2693321,0.2707829,0.3205946,0.3219939,0.3714757,0.3729162,0.4211923,0.4226227,0.4688781],"emit_ready_request_seconds":[2.8837394,2.9628074,2.9644813,3.0282688,3.0297538,3.0882369,3.089734,3.1460663,3.147462,3.1973847,3.1987114,3.2480684,3.249586,3.2978165,3.2993352,3.3456371],"emit_flush_request_seconds":[2.8845427,2.9636536,2.9653798,3.0292604,3.0307499,3.0892372,3.0907996,3.1471444,3.1485952,3.1984069,3.1998062,3.249288,3.2507285,3.2990046,3.300435,3.3466904],"cost_split":false,"cost_sync":false,"cost_phases_ms":{"initdraftctx":0,"draft":0,"verify":0,"hcopy":0,"replay":0,"rollback":0,"emit":0,"unattributed":0},"cost_sync_calls":0,"cost_sync_wait_ms":0}
```
`qwen35_mtp_txn_done`（照录）：
```
{"event":"qwen35_mtp_txn_done","rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","end_pos":3912,"max_new":256,"wall_s":0.59819}
{"event":"qwen35_mtp_txn_done","rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","end_pos":3912,"max_new":256,"wall_s":0.4691935}
```
`summary`（照录）：
```
{"event":"summary","exit":0,"prompt_tokens":3896,"output_tokens":32,"decode_evaluations":0,"prefill_seconds":4.0881237,"decode_seconds":3.9845221,"ttft_seconds":4.0960101,"prefill_tokens_per_s":953.004,"decode_evaluations_per_s":0,"strata_jobs":2493440,"strata_seconds":6.57992,"decode_layers":0,"decode_hits":3654,"decode_misses":6586,"prefill_layers":80,"prefill_entries":2493440,"cache_slots":2700,"cache_bytes":3354632192,"cache_mib":3199.2,"cache_payload_mib":3199.2,"cache_resident":2700,"hit_overlap":0,"hit_late":0,"hit_inflight":0,"all_hit_layers":0,"prefill_gpu_layers":80,"prefill_resident":4292,"prefill_streamed":14698,"prefill_product_calls":2460,"prefill_stream_mib":17626.91,"prefill_workspace_mib":539.37,"prefill_ring_mib":53.75,"prefill_loan_slots":479,"prefill_loan_mib":566.61,"prefill_owned_mib":0.00,"prefill_loan_restored":479,"prefill_loan_refill_ms":77.6,"prefill_loan_why":"","tier_ms":{"ids":163.8,"launch":81.3,"misses":416.1,"combine":16.6,"admit":0.0},"vram_samples":4,"peak_device_vram_mib":7577.3,"device_total_mib":8187.5,"prefill_process_seconds":4.0038373,"prefill_process_calls":1,"prefill_finish_seconds":0.0823802,"decode_eval_seconds":0,"decode_eval_spans":0,"decode_eval_per_s":0,"decode_observe_seconds":0,"decode_observe_rounds":0,"decode_first_vram_seconds":2.6e-06,"decode_first_logits_wait_seconds":8.3e-06,"decode_vram_seconds":2.6e-06,"decode_vram_samples":1,"vram_sample_seconds_total":0.0001533,"engine_perf_t_eval_ms_first":0,"engine_perf_t_eval_ms_last":0,"engine_perf_t_eval_ms_delta":0,"engine_perf_n_eval_first":1,"engine_perf_n_eval_last":1,"engine_perf_n_eval_delta":0,"engine_perf_t_p_eval_ms_first":0.038,"engine_perf_n_p_eval_first":3896,"shared_overlap_marker_layers":0,"shared_overlap_join_layers":0,"decode_scan_seconds":0.0004263,"decode_scan_rounds":1,"decode_sampler_seconds":0.0009388,"decode_sampler_rounds":1,"decode_topk_seconds":0,"decode_topk_rounds":0,"decode_piece_json_flush_seconds":0,"decode_piece_json_flush_rounds":0,"decode_piece_seconds":0,"decode_piece_rounds":0,"decode_piece_hex_seconds":0,"decode_piece_hex_rounds":0,"decode_token_write_seconds":0,"decode_token_write_rounds":0,"decode_flush_seconds":0,"decode_flush_rounds":0,"mtp_txn":true,"legacy_aggregate_authoritative":false}
```

## 3. 保真度对比表（A vs B）

### 3.1 逐请求核心判据

| 请求 | emitted A | emitted B | finish A | finish B | ids 序列 | 可见文本 | TPS_true_flush 差 B−A | prefill s A / B | prefill 差 |
|---|---|---|---|---|---|---|---|---|---|
| req1 | 16 | 16 | stop | stop | equal | equal | -9.983147 (-28.3%) | 3.853 / 4.088 | +0.235 |
| req2 | 16 | 16 | stop | stop | equal | equal | -1.036545 (-3.1%) | 2.836 / 2.876 | +0.040 |

| 请求 | prefill A | prefill B | Δ | TTFT A | TTFT B | Δ | TPS A | TPS B | Δ |
|---|---|---|---|---|---|---|---|---|---|
| req1 | 3.853 | 4.088 | +0.235 | 3.861 | 4.096 | +0.235 | 35.331270 | 25.348123 | -9.983147 |
| req2 | 2.836 | 2.876 | +0.040 | 2.842 | 2.885 | +0.042 | 33.493701 | 32.457156 | -1.036545 |

- ids 序列判据：`ids_equal=True`（req1/req2 均为 True）；`index:id:role` 连接序列的 SHA256：A req1/req2 与 B req1/req2 四者完全一致＝`f876b40d97dd5fa276309562faacac640257edb01e38dc65d8cf8d37395f39a8`（同一 16-token JSON 序列，跨请求亦同＝确定性重放）。
- TPOT_true_flush（ms/token）：req1 A=28.304 B=39.451；req2 A=29.856 B=30.810。TPS_true_flush=(n−1)/(末 flush−首 flush)，n=16，逐请求由 flush_arrays 数组复算（本表数值即复算值，原文数组见 §2）。
- finish 语义：两例 req1/req2 均 `eos=1`（真实 EOG，引擎 `stop="pending_eog"`，end_pos 3912），finish=**stop**；max_new=256 未被顶到（emitted=16）——**不是 length**。

### 3.2 进程级（summary）

| 字段 | A | B | 判定 |
|---|---|---|---|
| prompt tokens | 3896 | 3896 | equal |
| output tokens | 32 | 32 | equal |
| prefill tok/s | 1011.24 | 953.004 | diff |
| decode hits | 3654 | 3654 | equal |
| decode misses | 6586 | 6586 | equal |
| peak VRAM MiB | 7577.3 | 7577.3 | equal |
| cache resident slots | 2700 | 2700 | equal |
| cache MiB | 3199.2 | 3199.2 | equal |
| prefill s (summary) | 3.8526796 | 4.0881237 | diff |
| decode s (summary) | 3.7579166 | 3.9845221 | diff |
| TTFT s (summary) | 3.8609597 | 4.0960101 | diff |

| tier_ms | A | B | 判定 |
|---|---|---|---|
| ids | 160.7 | 163.8 | diff |
| launch | 51.5 | 81.3 | diff |
| misses | 320.3 | 416.1 | diff |
| combine | 11.0 | 16.6 | diff |
| admit | 0.0 | 0.0 | equal |

### 3.3 资源与采样（10 s 采样，3/3 样本 ×2）

| 指标 | A | B |
|---|---|---|
| 采样数 | 3 | 3 |
| AvailPhys 最低 MB | 3971.4 | 3283.7 |
| 页文件估算峰值 MB | 16026.3 | 16055.7 |
| 采样 VRAM 峰值 MiB | 7523.0 | 7523.0 |
| 采样 GPU util 峰值 pct | 93.0 | 86.0 |
| 引擎 peak_device_vram_mib | 7577.3 | 7577.3 |
| gpu-lock exit / child exit | 0 / 0 | 0 / 0 |
| native elapsed s | 14.188 | 14.266 |
| postfallback | `batch/postfallback-A.txt`（无残留进程，VRAM 回 1127 MiB） | `batch/postfallback-B.txt`（同） |

## 4. 结论（按预登记口径 a–d）

- **a) 加载与结束**：两例均加载成功、`bridge ready` 行逐字相同；到达真实 finish（EOS stop，16 tokens/请求×2 请求），exit 0；无 NaN/OOM/CUDA 错误，无事务回滚；marker 无保留。
- **b) 输出一致性（硬判据）**：可见 token id 序列 req1/req2 均 **逐字节相同**（`ids_equal=True`），可见文本相同，role（draft/bonus/corrective）构成相同；emitted/finish/eos 相同；hits=3654、misses=6586、cache_resident=2700、cache=3199.2 MiB、peak VRAM=7577.3 MiB 全部相同。导入保真在本次样本上成立。
- **c) 计时差异（区间，不给单点结论）**：TPS_true_flush A∈[33.493701, 35.331270]（A 内跨请求离散 5.2%），B∈[25.348123, 32.457156]（B 内 21.9%）；prefill A∈[2.836, 3.853] s，B∈[2.876, 4.088] s（Δ=+0.040/+0.235 s）；TTFT Δ=+0.042/+0.235 s；tier_ms 的 launch（51.5→81.3）与 misses（320.3→416.1）B 偏高。n=2 请求/例、单 native，属小样本非稳态；req2 TPS 差 -3.1%，req1 TPS 差 -28.3%（超出 B 自身两请求离散 21.9% 的观测水平），**不能由本次样本判为构建回归**；计时等价/差异裁决需更多重复（另行立项）。
- **d) 不一致最小复现**：无 token/文本/finish/资源不一致；唯一显式差异＝req1 的 TPS（B=25.348123 vs A=35.331270，Δ=-9.983147，-28.3%），最小复现＝同 argv/env 单 native 第 1 请求的 `qwen35_mtp_txn_request_timing.engine_wall_seconds`（A=0.431108 s vs B=0.598190 s，剔除 prefill/TTFT 后 n=15 flush 区间）。

## 5. 偏差与事实登记（需主管知悉）

1. **预登记基线口径不适用（事实）**：本任务 argv 的 `--tokens-file` 是 `results/20261012-qwen35-decode-quality157/fixtures/fixture-benign/prompt.tokens`（3896 tokens，本轮 summary `prompt_tokens=3896`），与 347 的 L-benign（3791 tokens）不是同一 fixture；且本轮实际形状＝16-token JSON + 真实 EOS（finish=stop），**不是** 256-token length 形状。347 的 22.35/26.64 与 length 语义均不作本任务比较基线（review 第 1 条的假设已修正）。
2. **fixture 文件哈希与自身 manifest 不一致（记录事实，不影响 A/B 保真）**：现盘 `prompt.tokens` sha256＝`4e6b2529aaab49d3c7a0b4f8f34bf5312b3c18dacbc688befd437421ad6a631d`，其 `fixture-manifest.json` 记 `tokens_sha256=2b9e0d52d0c6f33b0537166880008d6d85525787048b96c6735747a62e7c661a`；token 计数 3896 与 manifest `prompt_token_count=3896` 一致。两例消费的是同一文件，A/B 可比性不受影响。
3. **A 重跑登记**：attempt1（16:30:12–16:30:33）本体 exit 0、token 输出与 attempt2 一致（原件 `attempt1-A/`）；重跑原因＝我方 sampler 脚本参数 bug（已修复），非 exe/GPU 失败，无 fence、无预算外 native（3/≤4）。
4. **依赖位置**：新仓 exe 消费的 model/pack/profile/fixture 仍位于研究仓（cwd=D:/work/Strata魔改），与 354 记录的打包遗留项一致；本次未触发 cwd/绝对路径回退（预案未用）。
5. REJECTED 门未开（GU=0 已核：两例 raw 无 slice GU multi-token 行）；诊断门未额外启用；`禁用数字`未引用；EXPERIMENTAL（MTP K1/2700COMPACT）在本报告按实验口径书写，不作已认证。

## 6. 证据清单（`results/opencode-execution/strata-unified-smoke-355/`）

- `task.md`、`codex355-review-next.md`；`batch/start.json`、`batch/end.json`（批前后状态）、`batch/timecheck-A/B.txt`、`batch/postfallback-A/B.txt`、`batch/A/B-markercheck.txt`、`batch/A/B-result-note.txt`
- `job/job-A.json`、`job/job-B.json`；`audit/audit-A.json`、`audit/audit-B.json`；`result/result-A.json`、`result/result-B.json`、`result/A/B-ready.json`
- `raw/raw-A/child_00.log`、`raw/raw-B/child_00.log`（原生 stdout 全量）；`pre-env/pre-A.json`、`pre-env/pre-B.json`、`pre-env/pre-A/B-ramgate.json`
- `samples/samples-A.csv`、`samples/samples-B.csv`（10 s）、`samples/samples-A/B-summary.json`；`parse-A.json`、`parse-B.json`；`compare-AB.json`；`attempt1-A/`（A 第 1 次原件）
- 运行器：`orbit355.py`、`run355.sh`、`sampler355.ps1`（证据目录内）
