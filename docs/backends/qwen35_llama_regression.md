# Qwen35-llama 新构建计时回归对照（task 358，ABAB 配对；2026-10-14）

**一行状态**：4 native（A1→B1→B2→A2）全部 exit 0、无 fence、无重跑；8 请求守卫全为 emitted=16 / finish=stop / eos=1；warm（req2）TPS：A_med=33.061975、B_med=34.469589、Δ=+1.407614（+4.2575%）、漂移带 4.248466 → 按预登记规则 **PASS**（本次样本中新构建未见 warm 回归）。

## 1. 设计与执行事实

- 性质＝新/旧 exe 同机 ABAB 配对计时回归（356/357 两次自动终止后第三次执行）。预登记口径：warm＝各 native 的 req2、cold＝req1 分列；Δ=B_warm_median−A_warm_median；漂移带=max(|A1w−A2w|,|B1w−B2w|)；成对符号（A1 vs B1、A2 vs B2）。
- A1/A2＝旧实验 exe `results/phase2_mtp/consistentSlice300/strata.exe`（sha256 `03bd7068f284450cef90aa633bc248c72b7272ee143da9a54443d79db84dde5f`，139,107,840 B）；B1/B2＝新仓构建 `D:/work/StrataUnified/strata/backends/qwen35_llama/build/strata.exe`（sha256 `6062a714c38744a6adc1cb052f3242c304a3ff32d158545933834455544dc9b4`，139,208,192 B）。批前 sha256sum 复核与声明一致；gpu-lock 运行时二次校验（binary_sha256，全部 exit 0）。
- argv/env/cwd 与 355 job 原件逐项一致：`job/job-A1|A2|B1|B2.json` 的 commands 与 355 `job/job-A.json`/`job-B.json` 程序化比对**全等**；4 例 `pre-env/pre-*.json` 的 argv tail 互等、env_subset 互等、gates 互等；唯一差异＝argv[0]（A 组=旧 exe 路径，B 组=新 exe 路径）。tokens/profile 现盘 SHA 四例一致（`4e6b2529…` / `fab192ff…`）。
- 12 项 env gate 照 355 逐项设置：ROUTER_WEIGHTS_DEVICE=1、PACKED_IDS=1、EXPERT_ORDER=count_desc、COMPACT_SLOTS=1、STOP_EOG=1、DECODE_TIMESTAMPS=1、MTP_TXN=1、MTP_TXN_K=1、MTP_TXN_REQUESTS=2、SMALL_T_NATIVE=1、SMALL_T_MULTI=1、SLICE_GU_MULTI=0；REJECTED 门 `SLICE_GU_MULTI=0` 保持关；清 `STRATA_*`/`GGML_*`/`LLAMA_GRAPH*`/`CUDA_LAUNCH_BLOCKING` 后设置；PATH 前置 `runtime/cuda-12.8.1/bin`；child 去 `STRATA_OPENCODE_OBSERVER_LOG_DIR`。
- 单 GPU owner：`tools/gpu-lock.py`（sha256 `337426d63d5e36f199b8bca3e46bea041987a5d68272f426a5976f50da2a7111`）逐 native `run`；`max_launch=1`、`runtime_limit_seconds=300`、wait≤60 s。四例 wait attempts=1 / wait 0.0 s；child exit 0、timed_out=false；native elapsed=19.813 / 22.204 / 14.547 / 14.547 s；marker 均 auto-removed（`retained=false`）；批末（16:56:50）lock 1 B 空、无 marker、无 strata 进程、VRAM 1129 MiB、util 0%。
- RAM 门（pre-run avail≥8 GiB）：A1=17768.6、B1=17598.3、B2=17851.6、A2=17648.0 MB，全 pass。
- 时间窗：task.md mtime 16:47:58 起；四次 check 时点 elapsed=4.43 / 5.37 / 6.17 / 6.82 min（全 <20）；批末 8.87 min。
- 10 s 采样 CSV 行数：A1=4、B1=4、B2=3、A2=3；postfallback 四例均无残留进程。每 native 恰好执行 1 次（4/4），无重跑、无补跑。
- 未改动任何源码（新仓 backends/**、官方文件、研究仓源码一律只读）；新仓仅新增本文档；全部写操作仅限 358 证据目录与本文件。

## 2. 逐例原文行（verbatim，均出自各例 `raw/raw-*/child_00.log`）

### 2.1 A1（旧 exe 03bd7068）

READY 类启动行：
```
strata qwen35: bridge ready: 40 layers x 256 experts, 2048 x 512 experts top-8, workers 7, native tier on, prompt experts GPU/MMQ (grouped, resident cache + bounded cold ring)
```
prefill 段起始：
```
{"event":"prefill_start","since_process_start_seconds":8.4934951}
```
`qwen35_mtp_txn_request_timing`（照录）：
```
{"event":"qwen35_mtp_txn_request_timing","request":1,"anchor":"prefill_begin","reset_seconds":-1,"prefill_process_seconds":5.2311923,"prefill_finish_seconds":0.08028,"prefill_seconds":5.3135459,"seed_sample_seconds":5.3154115,"engine_entry_seconds":5.3159029,"ttft_flush_request_seconds":5.3232083,"eos_complete_request_seconds":5.7300558,"request_end_seconds":5.7349797,"json_closed":true,"json_close_flush_request_seconds":5.7298409,"engine_wall_seconds":0.4141529,"rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","eog_id":248046}
{"event":"qwen35_mtp_txn_request_timing","request":2,"anchor":"request_reset","reset_seconds":0.0014731,"prefill_process_seconds":2.7999528,"prefill_finish_seconds":0.0897551,"prefill_seconds":2.8923068,"seed_sample_seconds":2.8935418,"engine_entry_seconds":2.8943227,"ttft_flush_request_seconds":2.9028971,"eos_complete_request_seconds":3.3880305,"request_end_seconds":3.3932231,"json_closed":true,"json_close_flush_request_seconds":3.3877418,"engine_wall_seconds":0.4937078,"rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","eog_id":248046}
```
`qwen35_mtp_txn_request_flush_arrays`（照录，逐 token flush 时间数组）：
```
{"event":"qwen35_mtp_txn_request_flush_arrays","request":1,"engine_entry_seconds":5.3159029,"emit_ready_engine_seconds":[0.0065356,0.0897712,0.09292,0.1524068,0.1533736,0.1970803,0.1980489,0.2388862,0.2398023,0.2820344,0.2829556,0.3255814,0.3265878,0.3708045,0.3718397,0.4132424],"emit_flush_engine_seconds":[0.0073054,0.0904501,0.0936042,0.1531357,0.1540943,0.19777,0.1987726,0.2395732,0.2405269,0.2827175,0.2837025,0.3263581,0.3273214,0.3715835,0.3725883,0.413938],"emit_ready_request_seconds":[5.3224385,5.4056741,5.4088229,5.4683097,5.4692765,5.5129832,5.5139518,5.5547891,5.5557052,5.5979373,5.5988585,5.6414843,5.6424907,5.6867074,5.6877426,5.7291453],"emit_flush_request_seconds":[5.3232083,5.406353,5.4095071,5.4690386,5.4699972,5.5136729,5.5146755,5.5554761,5.5564298,5.5986204,5.5996054,5.642261,5.6432243,5.6874864,5.6884912,5.7298409],"cost_split":false,"cost_sync":false,"cost_phases_ms":{"initdraftctx":0,"draft":0,"verify":0,"hcopy":0,"replay":0,"rollback":0,"emit":0,"unattributed":0},"cost_sync_calls":0,"cost_sync_wait_ms":0}
{"event":"qwen35_mtp_txn_request_flush_arrays","request":2,"engine_entry_seconds":2.8943227,"emit_ready_engine_seconds":[0.0075467,0.1015062,0.1037127,0.1692639,0.1707771,0.2307965,0.2323268,0.2913681,0.2928481,0.3463788,0.3476222,0.3952118,0.3966255,0.4452863,0.4467338,0.4923785],"emit_flush_engine_seconds":[0.0085744,0.1024112,0.1046804,0.170298,0.1718245,0.2318242,0.2334074,0.2923551,0.2939087,0.3473282,0.3486633,0.3963307,0.3976778,0.4464223,0.4478087,0.4934191],"emit_ready_request_seconds":[2.9018694,2.9958289,2.9980354,3.0635866,3.0650998,3.1251192,3.1266495,3.1856908,3.1871708,3.2407015,3.2419449,3.2895345,3.2909482,3.339609,3.3410565,3.3867012],"emit_flush_request_seconds":[2.9028971,2.9967339,2.9990031,3.0646207,3.0661472,3.1261469,3.1277301,3.1866778,3.1882314,3.2416509,3.242986,3.2906534,3.2920005,3.340745,3.3421314,3.3877418],"cost_split":false,"cost_sync":false,"cost_phases_ms":{"initdraftctx":0,"draft":0,"verify":0,"hcopy":0,"replay":0,"rollback":0,"emit":0,"unattributed":0},"cost_sync_calls":0,"cost_sync_wait_ms":0}
```
`qwen35_mtp_txn_done`（照录）：
```
{"event":"qwen35_mtp_txn_done","rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","end_pos":3912,"max_new":256,"wall_s":0.4141529}
{"event":"qwen35_mtp_txn_done","rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","end_pos":3912,"max_new":256,"wall_s":0.4937078}
```
`summary`（照录）：
```
{"event":"summary","exit":0,"prompt_tokens":3896,"output_tokens":32,"decode_evaluations":0,"prefill_seconds":5.3135459,"decode_seconds":3.8368981,"ttft_seconds":5.3232083,"prefill_tokens_per_s":733.22,"decode_evaluations_per_s":0,"strata_jobs":2493440,"strata_seconds":6.9392,"decode_layers":0,"decode_hits":3654,"decode_misses":6586,"prefill_layers":80,"prefill_entries":2493440,"cache_slots":2700,"cache_bytes":3354632192,"cache_mib":3199.2,"cache_payload_mib":3199.2,"cache_resident":2700,"hit_overlap":0,"hit_late":0,"hit_inflight":0,"all_hit_layers":0,"prefill_gpu_layers":80,"prefill_resident":4292,"prefill_streamed":14698,"prefill_product_calls":2460,"prefill_stream_mib":17626.91,"prefill_workspace_mib":539.37,"prefill_ring_mib":53.75,"prefill_loan_slots":479,"prefill_loan_mib":566.61,"prefill_owned_mib":0.00,"prefill_loan_restored":479,"prefill_loan_refill_ms":82.7,"prefill_loan_why":"","tier_ms":{"ids":157.4,"launch":53.6,"misses":313.2,"combine":11.2,"admit":0.0},"vram_samples":4,"peak_device_vram_mib":7577.3,"device_total_mib":8187.5,"prefill_process_seconds":5.2311923,"prefill_process_calls":1,"prefill_finish_seconds":0.08028,"decode_eval_seconds":0,"decode_eval_spans":0,"decode_eval_per_s":0,"decode_observe_seconds":0,"decode_observe_rounds":0,"decode_first_vram_seconds":2.6e-06,"decode_first_logits_wait_seconds":8.4e-06,"decode_vram_seconds":2.6e-06,"decode_vram_samples":1,"vram_sample_seconds_total":0.0001535,"engine_perf_t_eval_ms_first":0,"engine_perf_t_eval_ms_last":0,"engine_perf_t_eval_ms_delta":0,"engine_perf_n_eval_first":1,"engine_perf_n_eval_last":1,"engine_perf_n_eval_delta":0,"engine_perf_t_p_eval_ms_first":0.019,"engine_perf_n_p_eval_first":3896,"shared_overlap_marker_layers":0,"shared_overlap_join_layers":0,"decode_scan_seconds":0.0004997,"decode_scan_rounds":1,"decode_sampler_seconds":0.0010926,"decode_sampler_rounds":1,"decode_topk_seconds":0,"decode_topk_rounds":0,"decode_piece_json_flush_seconds":0,"decode_piece_json_flush_rounds":0,"decode_piece_seconds":0,"decode_piece_rounds":0,"decode_piece_hex_seconds":0,"decode_piece_hex_rounds":0,"decode_token_write_seconds":0,"decode_token_write_rounds":0,"decode_flush_seconds":0,"decode_flush_rounds":0,"mtp_txn":true,"legacy_aggregate_authoritative":false}
```

### 2.2 B1（新 exe 6062a714）

READY 类启动行：
```
strata qwen35: bridge ready: 40 layers x 256 experts, 2048 x 512 experts top-8, workers 7, native tier on, prompt experts GPU/MMQ (grouped, resident cache + bounded cold ring)
```
prefill 段起始：
```
{"event":"prefill_start","since_process_start_seconds":4.2968588}
```
`qwen35_mtp_txn_request_timing`（照录）：
```
{"event":"qwen35_mtp_txn_request_timing","request":1,"anchor":"prefill_begin","reset_seconds":-1,"prefill_process_seconds":11.7085854,"prefill_finish_seconds":0.1229802,"prefill_seconds":11.8337171,"seed_sample_seconds":11.8355038,"engine_entry_seconds":11.836167,"ttft_flush_request_seconds":11.8434176,"eos_complete_request_seconds":12.4786421,"request_end_seconds":12.4836814,"json_closed":true,"json_close_flush_request_seconds":12.4782359,"engine_wall_seconds":0.6424751,"rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","eog_id":248046}
{"event":"qwen35_mtp_txn_request_timing","request":2,"anchor":"request_reset","reset_seconds":0.0017638,"prefill_process_seconds":2.7777439,"prefill_finish_seconds":0.087439,"prefill_seconds":2.86842,"seed_sample_seconds":2.8692661,"engine_entry_seconds":2.8699361,"ttft_flush_request_seconds":2.8760737,"eos_complete_request_seconds":3.3154281,"request_end_seconds":3.3196245,"json_closed":true,"json_close_flush_request_seconds":3.3151409,"engine_wall_seconds":0.445492,"rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","eog_id":248046}
```
`qwen35_mtp_txn_request_flush_arrays`（照录，逐 token flush 时间数组）：
```
{"event":"qwen35_mtp_txn_request_flush_arrays","request":1,"engine_entry_seconds":11.836167,"emit_ready_engine_seconds":[0.0063974,0.1043946,0.1082,0.1988719,0.2004388,0.2964175,0.2979163,0.3618721,0.363381,0.4272466,0.42877,0.49837,0.5002306,0.5774317,0.5793075,0.6408956],"emit_flush_engine_seconds":[0.0072506,0.1053371,0.1091756,0.1999322,0.2015074,0.2974009,0.2989425,0.362868,0.3645067,0.4282418,0.4298586,0.4996862,0.5013473,0.578692,0.580503,0.6420689],"emit_ready_request_seconds":[11.8425644,11.9405616,11.944367,12.0350389,12.0366058,12.1325845,12.1340833,12.1980391,12.199548,12.2634136,12.264937,12.334537,12.3363976,12.4135987,12.4154745,12.4770626],"emit_flush_request_seconds":[11.8434176,11.9415041,11.9453426,12.0360992,12.0376744,12.1335679,12.1351095,12.199035,12.2006737,12.2644088,12.2660256,12.3358532,12.3375143,12.414859,12.41667,12.4782359],"cost_split":false,"cost_sync":false,"cost_phases_ms":{"initdraftctx":0,"draft":0,"verify":0,"hcopy":0,"replay":0,"rollback":0,"emit":0,"unattributed":0},"cost_sync_calls":0,"cost_sync_wait_ms":0}
{"event":"qwen35_mtp_txn_request_flush_arrays","request":2,"engine_entry_seconds":2.8699361,"emit_ready_engine_seconds":[0.0052758,0.0694773,0.0710838,0.1299917,0.131324,0.1821427,0.1836155,0.2350291,0.2364279,0.2859613,0.2872786,0.3452447,0.346991,0.3971565,0.3986077,0.4441715],"emit_flush_engine_seconds":[0.0061376,0.0704363,0.0720595,0.1310328,0.1323501,0.1832595,0.1847363,0.2361271,0.2375516,0.2869876,0.288387,0.34647,0.3481581,0.3983302,0.3997066,0.4452048],"emit_ready_request_seconds":[2.8752119,2.9394134,2.9410199,2.9999278,3.0012601,3.0520788,3.0535516,3.1049652,3.106364,3.1558974,3.1572147,3.2151808,3.2169271,3.2670926,3.2685438,3.3141076],"emit_flush_request_seconds":[2.8760737,2.9403724,2.9419956,3.0009689,3.0022862,3.0531956,3.0546724,3.1060632,3.1074877,3.1569237,3.1583231,3.2164061,3.2180942,3.2682663,3.2696427,3.3151409],"cost_split":false,"cost_sync":false,"cost_phases_ms":{"initdraftctx":0,"draft":0,"verify":0,"hcopy":0,"replay":0,"rollback":0,"emit":0,"unattributed":0},"cost_sync_calls":0,"cost_sync_wait_ms":0}
```
`qwen35_mtp_txn_done`（照录）：
```
{"event":"qwen35_mtp_txn_done","rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","end_pos":3912,"max_new":256,"wall_s":0.6424751}
{"event":"qwen35_mtp_txn_done","rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","end_pos":3912,"max_new":256,"wall_s":0.445492}
```
`summary`（照录）：
```
{"event":"summary","exit":0,"prompt_tokens":3896,"output_tokens":32,"decode_evaluations":0,"prefill_seconds":11.8337171,"decode_seconds":3.9972647,"ttft_seconds":11.8434176,"prefill_tokens_per_s":329.229,"decode_evaluations_per_s":0,"strata_jobs":2493440,"strata_seconds":14.2363,"decode_layers":0,"decode_hits":3654,"decode_misses":6586,"prefill_layers":80,"prefill_entries":2493440,"cache_slots":2700,"cache_bytes":3354632192,"cache_mib":3199.2,"cache_payload_mib":3199.2,"cache_resident":2700,"hit_overlap":0,"hit_late":0,"hit_inflight":0,"all_hit_layers":0,"prefill_gpu_layers":80,"prefill_resident":4292,"prefill_streamed":14698,"prefill_product_calls":2460,"prefill_stream_mib":17626.91,"prefill_workspace_mib":539.37,"prefill_ring_mib":53.75,"prefill_loan_slots":479,"prefill_loan_mib":566.61,"prefill_owned_mib":0.00,"prefill_loan_restored":479,"prefill_loan_refill_ms":80.3,"prefill_loan_why":"","tier_ms":{"ids":181.5,"launch":65.9,"misses":431.8,"combine":14.4,"admit":0.0},"vram_samples":4,"peak_device_vram_mib":7577.3,"device_total_mib":8187.5,"prefill_process_seconds":11.7085854,"prefill_process_calls":1,"prefill_finish_seconds":0.1229802,"decode_eval_seconds":0,"decode_eval_spans":0,"decode_eval_per_s":0,"decode_observe_seconds":0,"decode_observe_rounds":0,"decode_first_vram_seconds":1.89e-05,"decode_first_logits_wait_seconds":1.09e-05,"decode_vram_seconds":1.89e-05,"decode_vram_samples":1,"vram_sample_seconds_total":0.0001778,"engine_perf_t_eval_ms_first":0,"engine_perf_t_eval_ms_last":0,"engine_perf_t_eval_ms_delta":0,"engine_perf_n_eval_first":1,"engine_perf_n_eval_last":1,"engine_perf_n_eval_delta":0,"engine_perf_t_p_eval_ms_first":0.019,"engine_perf_n_p_eval_first":3896,"shared_overlap_marker_layers":0,"shared_overlap_join_layers":0,"decode_scan_seconds":0.000532,"decode_scan_rounds":1,"decode_sampler_seconds":0.0008773,"decode_sampler_rounds":1,"decode_topk_seconds":0,"decode_topk_rounds":0,"decode_piece_json_flush_seconds":0,"decode_piece_json_flush_rounds":0,"decode_piece_seconds":0,"decode_piece_rounds":0,"decode_piece_hex_seconds":0,"decode_piece_hex_rounds":0,"decode_token_write_seconds":0,"decode_token_write_rounds":0,"decode_flush_seconds":0,"decode_flush_rounds":0,"mtp_txn":true,"legacy_aggregate_authoritative":false}
```

### 2.3 B2（新 exe 6062a714）

READY 类启动行：
```
strata qwen35: bridge ready: 40 layers x 256 experts, 2048 x 512 experts top-8, workers 7, native tier on, prompt experts GPU/MMQ (grouped, resident cache + bounded cold ring)
```
prefill 段起始：
```
{"event":"prefill_start","since_process_start_seconds":4.2545384}
```
`qwen35_mtp_txn_request_timing`（照录）：
```
{"event":"qwen35_mtp_txn_request_timing","request":1,"anchor":"prefill_begin","reset_seconds":-1,"prefill_process_seconds":4.3456633,"prefill_finish_seconds":0.0842819,"prefill_seconds":4.4338657,"seed_sample_seconds":4.4356689,"engine_entry_seconds":4.4363149,"ttft_flush_request_seconds":4.443079,"eos_complete_request_seconds":4.9654726,"request_end_seconds":4.970611,"json_closed":true,"json_close_flush_request_seconds":4.9649177,"engine_wall_seconds":0.5291577,"rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","eog_id":248046}
{"event":"qwen35_mtp_txn_request_timing","request":2,"anchor":"request_reset","reset_seconds":0.00157,"prefill_process_seconds":2.7817927,"prefill_finish_seconds":0.0845018,"prefill_seconds":2.8692755,"seed_sample_seconds":2.8703105,"engine_entry_seconds":2.870943,"ttft_flush_request_seconds":2.8774086,"eos_complete_request_seconds":3.3090009,"request_end_seconds":3.3137562,"json_closed":true,"json_close_flush_request_seconds":3.3087425,"engine_wall_seconds":0.4380579,"rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","eog_id":248046}
```
`qwen35_mtp_txn_request_flush_arrays`（照录，逐 token flush 时间数组）：
```
{"event":"qwen35_mtp_txn_request_flush_arrays","request":1,"engine_entry_seconds":4.4363149,"emit_ready_engine_seconds":[0.0059375,0.0795849,0.0814017,0.1548327,0.1563449,0.2140653,0.2156137,0.2729315,0.2746066,0.3364134,0.3380577,0.4008969,0.4027182,0.4666859,0.46847,0.5266344],"emit_flush_engine_seconds":[0.0067641,0.0804501,0.0823187,0.1558537,0.157338,0.2151119,0.2167854,0.2740688,0.2758122,0.3375431,0.3392957,0.4021975,0.4039684,0.4678762,0.469728,0.5286028],"emit_ready_request_seconds":[4.4422524,4.5158998,4.5177166,4.5911476,4.5926598,4.6503802,4.6519286,4.7092464,4.7109215,4.7727283,4.7743726,4.8372118,4.8390331,4.9030008,4.9047849,4.9629493],"emit_flush_request_seconds":[4.443079,4.516765,4.5186336,4.5921686,4.5936529,4.6514268,4.6531003,4.7103837,4.7121271,4.773858,4.7756106,4.8385124,4.8402833,4.9041911,4.9060429,4.9649177],"cost_split":false,"cost_sync":false,"cost_phases_ms":{"initdraftctx":0,"draft":0,"verify":0,"hcopy":0,"replay":0,"rollback":0,"emit":0,"unattributed":0},"cost_sync_calls":0,"cost_sync_wait_ms":0}
{"event":"qwen35_mtp_txn_request_flush_arrays","request":2,"engine_entry_seconds":2.870943,"emit_ready_engine_seconds":[0.0056496,0.0758906,0.077391,0.1348141,0.136161,0.1874,0.1886885,0.2381857,0.2394324,0.2883698,0.2897632,0.3402278,0.3417427,0.3908652,0.3923222,0.4367678],"emit_flush_engine_seconds":[0.0064656,0.0767706,0.0782892,0.1358712,0.1371876,0.1884162,0.1897537,0.2391725,0.2404756,0.2894533,0.2908788,0.3414348,0.3428772,0.3920464,0.3934613,0.4377995],"emit_ready_request_seconds":[2.8765926,2.9468336,2.948334,3.0057571,3.007104,3.058343,3.0596315,3.1091287,3.1103754,3.1593128,3.1607062,3.2111708,3.2126857,3.2618082,3.2632652,3.3077108],"emit_flush_request_seconds":[2.8774086,2.9477136,2.9492322,3.0068142,3.0081306,3.0593592,3.0606967,3.1101155,3.1114186,3.1603963,3.1618218,3.2123778,3.2138202,3.2629894,3.2644043,3.3087425],"cost_split":false,"cost_sync":false,"cost_phases_ms":{"initdraftctx":0,"draft":0,"verify":0,"hcopy":0,"replay":0,"rollback":0,"emit":0,"unattributed":0},"cost_sync_calls":0,"cost_sync_wait_ms":0}
```
`qwen35_mtp_txn_done`（照录）：
```
{"event":"qwen35_mtp_txn_done","rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","end_pos":3912,"max_new":256,"wall_s":0.5291577}
{"event":"qwen35_mtp_txn_done","rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","end_pos":3912,"max_new":256,"wall_s":0.4380579}
```
`summary`（照录）：
```
{"event":"summary","exit":0,"prompt_tokens":3896,"output_tokens":32,"decode_evaluations":0,"prefill_seconds":4.4338657,"decode_seconds":3.8781099,"ttft_seconds":4.443079,"prefill_tokens_per_s":878.691,"decode_evaluations_per_s":0,"strata_jobs":2493440,"strata_seconds":6.52339,"decode_layers":0,"decode_hits":3654,"decode_misses":6586,"prefill_layers":80,"prefill_entries":2493440,"cache_slots":2700,"cache_bytes":3354632192,"cache_mib":3199.2,"cache_payload_mib":3199.2,"cache_resident":2700,"hit_overlap":0,"hit_late":0,"hit_inflight":0,"all_hit_layers":0,"prefill_gpu_layers":80,"prefill_resident":4292,"prefill_streamed":14698,"prefill_product_calls":2460,"prefill_stream_mib":17626.91,"prefill_workspace_mib":539.37,"prefill_ring_mib":53.75,"prefill_loan_slots":479,"prefill_loan_mib":566.61,"prefill_owned_mib":0.00,"prefill_loan_restored":479,"prefill_loan_refill_ms":77.4,"prefill_loan_why":"","tier_ms":{"ids":157.4,"launch":70.2,"misses":350.2,"combine":14.2,"admit":0.0},"vram_samples":4,"peak_device_vram_mib":7577.3,"device_total_mib":8187.5,"prefill_process_seconds":4.3456633,"prefill_process_calls":1,"prefill_finish_seconds":0.0842819,"decode_eval_seconds":0,"decode_eval_spans":0,"decode_eval_per_s":0,"decode_observe_seconds":0,"decode_observe_rounds":0,"decode_first_vram_seconds":3.3e-06,"decode_first_logits_wait_seconds":9.8e-06,"decode_vram_seconds":3.3e-06,"decode_vram_samples":1,"vram_sample_seconds_total":0.0001534,"engine_perf_t_eval_ms_first":0,"engine_perf_t_eval_ms_last":0,"engine_perf_t_eval_ms_delta":0,"engine_perf_n_eval_first":1,"engine_perf_n_eval_last":1,"engine_perf_n_eval_delta":0,"engine_perf_t_p_eval_ms_first":0.037,"engine_perf_n_p_eval_first":3896,"shared_overlap_marker_layers":0,"shared_overlap_join_layers":0,"decode_scan_seconds":0.0005329,"decode_scan_rounds":1,"decode_sampler_seconds":0.0011315,"decode_sampler_rounds":1,"decode_topk_seconds":0,"decode_topk_rounds":0,"decode_piece_json_flush_seconds":0,"decode_piece_json_flush_rounds":0,"decode_piece_seconds":0,"decode_piece_rounds":0,"decode_piece_hex_seconds":0,"decode_piece_hex_rounds":0,"decode_token_write_seconds":0,"decode_token_write_rounds":0,"decode_flush_seconds":0,"decode_flush_rounds":0,"mtp_txn":true,"legacy_aggregate_authoritative":false}
```

### 2.4 A2（旧 exe 03bd7068）

READY 类启动行：
```
strata qwen35: bridge ready: 40 layers x 256 experts, 2048 x 512 experts top-8, workers 7, native tier on, prompt experts GPU/MMQ (grouped, resident cache + bounded cold ring)
```
prefill 段起始：
```
{"event":"prefill_start","since_process_start_seconds":4.251957}
```
`qwen35_mtp_txn_request_timing`（照录）：
```
{"event":"qwen35_mtp_txn_request_timing","request":1,"anchor":"prefill_begin","reset_seconds":-1,"prefill_process_seconds":4.3576392,"prefill_finish_seconds":0.0853434,"prefill_seconds":4.4467236,"seed_sample_seconds":4.4484356,"engine_entry_seconds":4.4490986,"ttft_flush_request_seconds":4.4566884,"eos_complete_request_seconds":4.9801846,"request_end_seconds":4.9851893,"json_closed":true,"json_close_flush_request_seconds":4.9798236,"engine_wall_seconds":0.531086,"rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","eog_id":248046}
{"event":"qwen35_mtp_txn_request_timing","request":2,"anchor":"request_reset","reset_seconds":0.0015853,"prefill_process_seconds":2.8040971,"prefill_finish_seconds":0.0851894,"prefill_seconds":2.8922281,"seed_sample_seconds":2.8932066,"engine_entry_seconds":2.8938219,"ttft_flush_request_seconds":2.9002216,"eos_complete_request_seconds":3.3268618,"request_end_seconds":3.3322814,"json_closed":true,"json_close_flush_request_seconds":3.326525,"engine_wall_seconds":0.4330399,"rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","eog_id":248046}
```
`qwen35_mtp_txn_request_flush_arrays`（照录，逐 token flush 时间数组）：
```
{"event":"qwen35_mtp_txn_request_flush_arrays","request":1,"engine_entry_seconds":4.4490986,"emit_ready_engine_seconds":[0.0066986,0.0934791,0.0954475,0.1780551,0.1795466,0.2404866,0.2423715,0.3015319,0.3030235,0.3612081,0.3629264,0.4226472,0.4244001,0.4813333,0.48299,0.5296567],"emit_flush_engine_seconds":[0.0075898,0.0944305,0.0964675,0.1790616,0.1806282,0.2416641,0.2434301,0.3025467,0.304103,0.3623925,0.3640813,0.4238997,0.4255201,0.4825951,0.4841365,0.530725],"emit_ready_request_seconds":[4.4557972,4.5425777,4.5445461,4.6271537,4.6286452,4.6895852,4.6914701,4.7506305,4.7521221,4.8103067,4.812025,4.8717458,4.8734987,4.9304319,4.9320886,4.9787553],"emit_flush_request_seconds":[4.4566884,4.5435291,4.5455661,4.6281602,4.6297268,4.6907627,4.6925287,4.7516453,4.7532016,4.8114911,4.8131799,4.8729983,4.8746187,4.9316937,4.9332351,4.9798236],"cost_split":false,"cost_sync":false,"cost_phases_ms":{"initdraftctx":0,"draft":0,"verify":0,"hcopy":0,"replay":0,"rollback":0,"emit":0,"unattributed":0},"cost_sync_calls":0,"cost_sync_wait_ms":0}
{"event":"qwen35_mtp_txn_request_flush_arrays","request":2,"engine_entry_seconds":2.8938219,"emit_ready_engine_seconds":[0.0055721,0.0698407,0.0713131,0.1282944,0.1295727,0.182381,0.1837207,0.2333619,0.2346297,0.2827266,0.2839404,0.3339811,0.3354111,0.3846424,0.3861008,0.4317148],"emit_flush_engine_seconds":[0.0063997,0.0707199,0.0722085,0.1293103,0.1305787,0.1833836,0.1847541,0.2343421,0.2356686,0.2836808,0.2850071,0.3351403,0.3365165,0.3857501,0.3871406,0.4327031],"emit_ready_request_seconds":[2.899394,2.9636626,2.965135,3.0221163,3.0233946,3.0762029,3.0775426,3.1271838,3.1284516,3.1765485,3.1777623,3.227803,3.229233,3.2784643,3.2799227,3.3255367],"emit_flush_request_seconds":[2.9002216,2.9645418,2.9660304,3.0231322,3.0244006,3.0772055,3.078576,3.128164,3.1294905,3.1775027,3.178829,3.2289622,3.2303384,3.279572,3.2809625,3.326525],"cost_split":false,"cost_sync":false,"cost_phases_ms":{"initdraftctx":0,"draft":0,"verify":0,"hcopy":0,"replay":0,"rollback":0,"emit":0,"unattributed":0},"cost_sync_calls":0,"cost_sync_wait_ms":0}
```
`qwen35_mtp_txn_done`（照录）：
```
{"event":"qwen35_mtp_txn_done","rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","end_pos":3912,"max_new":256,"wall_s":0.531086}
{"event":"qwen35_mtp_txn_done","rounds":8,"emitted":16,"eos":1,"stop":"pending_eog","end_pos":3912,"max_new":256,"wall_s":0.4330399}
```
`summary`（照录）：
```
{"event":"summary","exit":0,"prompt_tokens":3896,"output_tokens":32,"decode_evaluations":0,"prefill_seconds":4.4467236,"decode_seconds":3.8980606,"ttft_seconds":4.4566884,"prefill_tokens_per_s":876.151,"decode_evaluations_per_s":0,"strata_jobs":2493440,"strata_seconds":6.59367,"decode_layers":0,"decode_hits":3654,"decode_misses":6586,"prefill_layers":80,"prefill_entries":2493440,"cache_slots":2700,"cache_bytes":3354632192,"cache_mib":3199.2,"cache_payload_mib":3199.2,"cache_resident":2700,"hit_overlap":0,"hit_late":0,"hit_inflight":0,"all_hit_layers":0,"prefill_gpu_layers":80,"prefill_resident":4292,"prefill_streamed":14698,"prefill_product_calls":2460,"prefill_stream_mib":17626.91,"prefill_workspace_mib":539.37,"prefill_ring_mib":53.75,"prefill_loan_slots":479,"prefill_loan_mib":566.61,"prefill_owned_mib":0.00,"prefill_loan_restored":479,"prefill_loan_refill_ms":78.0,"prefill_loan_why":"","tier_ms":{"ids":161.9,"launch":64.0,"misses":338.7,"combine":13.6,"admit":0.0},"vram_samples":4,"peak_device_vram_mib":7577.3,"device_total_mib":8187.5,"prefill_process_seconds":4.3576392,"prefill_process_calls":1,"prefill_finish_seconds":0.0853434,"decode_eval_seconds":0,"decode_eval_spans":0,"decode_eval_per_s":0,"decode_observe_seconds":0,"decode_observe_rounds":0,"decode_first_vram_seconds":3.3e-06,"decode_first_logits_wait_seconds":9.5e-06,"decode_vram_seconds":3.3e-06,"decode_vram_samples":1,"vram_sample_seconds_total":0.0001589,"engine_perf_t_eval_ms_first":0,"engine_perf_t_eval_ms_last":0,"engine_perf_t_eval_ms_delta":0,"engine_perf_n_eval_first":1,"engine_perf_n_eval_last":1,"engine_perf_n_eval_delta":0,"engine_perf_t_p_eval_ms_first":0.026,"engine_perf_n_p_eval_first":3896,"shared_overlap_marker_layers":0,"shared_overlap_join_layers":0,"decode_scan_seconds":0.0005274,"decode_scan_rounds":1,"decode_sampler_seconds":0.0010417,"decode_sampler_rounds":1,"decode_topk_seconds":0,"decode_topk_rounds":0,"decode_piece_json_flush_seconds":0,"decode_piece_json_flush_rounds":0,"decode_piece_seconds":0,"decode_piece_rounds":0,"decode_piece_hex_seconds":0,"decode_piece_hex_rounds":0,"decode_token_write_seconds":0,"decode_token_write_rounds":0,"decode_flush_seconds":0,"decode_flush_rounds":0,"mtp_txn":true,"legacy_aggregate_authoritative":false}
```

## 3. 统计与对照

### 3.1 预登记 warm/cold 统计（TPS_true_flush；复算自 `emit_flush_request_seconds` 数组，(n−1)/(末−首)，8/8 与 parse 一致到 1e-6）

| 口径 | A1 | A2 | A 中位 | A 极差 | B1 | B2 | B 中位 | B 极差 |
|---|---|---|---|---|---|---|---|---|
| warm req2 | 30.937742 | 35.186208 | 33.061975 | 4.248466 | 34.163335 | 34.775843 | 34.469589 | 0.612508 |
| cold req1 | 36.888336 | 28.673276 | 32.780806 | 8.215060 | 23.628808 | 28.744514 | 26.186661 | 5.115706 |

warm 判定计算过程：
- Δ = B_med − A_med = 34.469589 − 33.061975 = **+1.407614**（= +4.2575% × A_med）
- 漂移带 = max(|A1w−A2w|, |B1w−B2w|) = max(4.248466, 0.612508) = **4.248466**
- 5% × A_med = 1.653099；阈值 = max(5% × A_med, 漂移带) = **4.248466**
- |Δ| = 1.407614 ≤ 4.248466 ✓；成对符号：B1−A1 = **+3.225593**、B2−A2 = **−0.410365** → 并非两对同负 ✓
- **判定：PASS**（预登记 FAIL 分支未触发：未同时满足 |Δ|>5% 且 |Δ|>漂移带 且两对同负）
- cold 仅登记不判定：Δ_cold = −6.594145（−20.1%），其中 B1 req1 为冷启动抖动（prefill 11.834 s，见 §5-1），n=2。

### 3.2 逐请求明细（4 例 × req1/req2）

| case | req | emitted | finish | eos | rounds | prefill s | TTFT s | EOS-complete s | engine_wall s | TPS_true_flush | TPOT_true_flush (s) | n_flush |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 1 | 16 | stop | 1 | 8 | 5.3135459 | 5.3232083 | 5.7300558 | 0.4141529 | 36.888336 | 0.027109 | 16 |
| A1 | 2 | 16 | stop | 1 | 8 | 2.8923068 | 2.9028971 | 3.3880305 | 0.4937078 | 30.937742 | 0.032323 | 16 |
| B1 | 1 | 16 | stop | 1 | 8 | 11.8337171 | 11.8434176 | 12.4786421 | 0.6424751 | 23.628808 | 0.042321 | 16 |
| B1 | 2 | 16 | stop | 1 | 8 | 2.8684200 | 2.8760737 | 3.3154281 | 0.4454920 | 34.163335 | 0.029271 | 16 |
| B2 | 1 | 16 | stop | 1 | 8 | 4.4338657 | 4.4430790 | 4.9654726 | 0.5291577 | 28.744514 | 0.034789 | 16 |
| B2 | 2 | 16 | stop | 1 | 8 | 2.8692755 | 2.8774086 | 3.3090009 | 0.4380579 | 34.775843 | 0.028756 | 16 |
| A2 | 1 | 16 | stop | 1 | 8 | 4.4467236 | 4.4566884 | 4.9801846 | 0.5310860 | 28.673276 | 0.034876 | 16 |
| A2 | 2 | 16 | stop | 1 | 8 | 2.8922281 | 2.9002216 | 3.3268618 | 0.4330399 | 35.186208 | 0.028420 | 16 |

- 守卫：8/8 emitted=16、finish=stop、eos=1（真实 EOG，`stop="pending_eog"`, `eog_id=248046`）；max_new=256 未被顶到 → 非 length。
- 保真：8/8 可见 token id 序列 SHA 同为 `f876b40d97dd5fa276309562faacac640257edb01e38dc65d8cf8d37395f39a8`；可见文本同为 `{"risk": 0, "violation": false}`（16-token JSON）；`json_closed=true`。
- warm 口径：req2 prefill 四例 2.868–2.892 s（离散很小）；差异主要来自 engine_wall/decode 段。

### 3.3 进程级 summary 对照

全等字段（4/4 equal）：prompt_tokens=3896、output_tokens=32、decode_hits=3654、decode_misses=6586、cache_resident=2700、cache_mib=3199.2、peak_device_vram_mib=7577.3、strata_jobs=2493440、prefill_process_calls=1。

| 字段 | A1 | B1 | B2 | A2 |
|---|---|---|---|---|
| prefill_seconds | 5.3135459 | 11.8337171 | 4.4338657 | 4.4467236 |
| decode_seconds | 3.8368981 | 3.9972647 | 3.8781099 | 3.8980606 |
| ttft_seconds | 5.3232083 | 11.8434176 | 4.4430790 | 4.4566884 |
| prefill_tokens_per_s | 733.22 | 329.229 | 878.691 | 876.151 |
| strata_seconds | 6.93920 | 14.23630 | 6.52339 | 6.59367 |

### 3.4 tier_ms 五项差（同 warm 口径：每例进程级值 + 两侧中位差）

| tier_ms | A1 | A2 | B1 | B2 | A_med | B_med | B_med−A_med |
|---|---|---|---|---|---|---|---|
| ids | 157.4 | 161.9 | 181.5 | 157.4 | 159.65 | 169.45 | +9.80 |
| launch | 53.6 | 64.0 | 65.9 | 70.2 | 58.80 | 68.05 | +9.25 |
| misses | 313.2 | 338.7 | 431.8 | 350.2 | 325.95 | 391.00 | +65.05 |
| combine | 11.2 | 13.6 | 14.4 | 14.2 | 12.40 | 14.30 | +1.90 |
| admit | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.00 |

注：B 侧中位五项中四项偏高（ids +9.80、launch +9.25、misses +65.05、combine +1.90 ms），与 355 观测同向；但对应 warm Δ 仍在漂移带内，按预登记规则不构成回归判定。

### 3.5 资源、租约与采样

| 指标 | A1 | B1 | B2 | A2 |
|---|---|---|---|---|
| RAM 门 avail MB / pass | 17768.6 / true | 17598.3 / true | 17851.6 / true | 17648.0 / true |
| 采样数（10 s） | 4 | 4 | 3 | 3 |
| 采样 avail 最低 MB | 2152.8 | 1431.2 | 3527.4 | 3806.8 |
| 页文件估算峰值 MB | 18051.3 | 18489.6 | 15985.9 | 15970.4 |
| 采样 VRAM 峰值 MiB | 6533.0 | 7539.0 | 7523.0 | 7523.0 |
| 采样 GPU util 峰值 % | 38 | 99 | 92 | 88 |
| 引擎 peak_device_vram_mib | 7577.3 | 7577.3 | 7577.3 | 7577.3 |
| gpu-lock wait attempts / s | 1 / 0.0 | 1 / 0.0 | 1 / 0.0 | 1 / 0.0 |
| native elapsed s | 19.813 | 22.204 | 14.547 | 14.547 |
| gpu-lock / child exit | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| marker retained | false | false | false | false |
| postfallback | 无残留 | 无残留 | 无残留 | 无残留 |

## 4. 判定与结论（按预登记口径）

- **PASS**（warm）：Δ=+4.2575% 在阈值（max(5%×A_med, 漂移带)=4.248466 TPS）之内，且成对符号不同向（+3.23 / −0.41）→ 本次样本中新构建 **未见 warm 慢速回归**；对应 FAIL 必述项（慢 X% + 具体 tier_ms 项）不适用。
- 保真硬判据全过：ids 序列 8/8 逐字节相同、文本相同、finish=stop（真实 EOS）、hits/misses/cache/peak VRAM 全等 → 回归样本内行为一致。
- 边界与口径：本判为**同机、同 argv/env、旧/新 exe 的 warm 计时对照**（EXPERIMENTAL：MTP K1 / 2700COMPACT），不是产品速度结论、不是跨量化比较；tier_ms B 侧偏高为事实登记，warm 差异的最终裁决需更多重复（另行立项）。
- cold（req1）不参与判定：B1 req1 冷启动抖动已登记（§5-1）。

## 5. 偏差与事实登记

1. **B1 req1 冷启动 outlier**：prefill=11.834 s（其余三例 4.43–5.31 s），对应 cold TPS 23.63（B 组另一例 28.74）。warm 口径不受影响（req2 prefill 四例 2.868–2.892 s）。
2. **小样本限制**：每 exe 仅 n=2 warm（native 各 1 次）。A 组 warm 内部漂移 4.248466 TPS（A1 30.94 vs A2 35.19）大于 Δ 本身，阈值实际由漂移带主导 → PASS 的强度受限于样本量，登记待重复。
3. **采样口径**：10 s 周期；A1 采样 VRAM 峰值仅捕获 6533.0 MiB（间隔误差），非采样口径以引擎 `peak_device_vram_mib=7577.3` 为准（四例全等）。
4. **fixture 事实沿用 355 登记**：`prompt.tokens` 现盘 sha256=`4e6b2529…` 与自身 manifest 记录不一致；本批 4 例消费同一文件、同一 SHA（§1），A/B 可比性不受影响。
5. 无 fence、无重跑（4/4 每 native 恰好一次）；A1→B1→B2→A2 严格串行；批末 elapsed 8.87 min <20；A1..A2 原始 raw/pre-env/result/audit 独立目录、原始 state 未被覆盖。
6. REJECTED 门未开（`SLICE_GU_MULTI=0`）；禁用数字未引用；EXPERIMENTAL 未写成已认证；未改动生产/官方/源码文件。
7. **356/357 失败教训已规避**：本批全部写操作仅限 `results/opencode-execution/strata-unified-regression-358/**` 与本仓 `docs/backends/qwen35_llama_regression.md`；未写 /tmp、%TEMP%、白名单外路径。

## 6. 证据清单（`results/opencode-execution/strata-unified-regression-358/`）

- job：`job/job-A1.json`、`job-A2.json`、`job-B1.json`、`job-B2.json`（355 原件 job-A/B.json 同时保留）
- 租约与结果：`audit/audit-*.json`、`result/result-*.json`、`result/*-ready.json`
- 原文：`raw/raw-{A1,B1,B2,A2}/child_00.log`（全量）；`pre-env/pre-*.json`、`pre-env/pre-*-ramgate.json`
- 采样与批记：`samples/samples-*.csv`、`samples/*-summary.json`、`batch/timecheck-*.txt`、`batch/postfallback-*.txt`、`batch/*-markercheck.txt`、`batch/*-result-note.txt`、`batch/end.json`
- 解析与统计：`parse-{A1,B1,B2,A2}.json`、`stats-358.json`、`stats358.py`
- harness 参数化：`adaptive-358.md`、`batch/diff-run358.txt`、`batch/diff-orbit358.txt`、`harness-copy-sha358.txt`
- 运行器：`run358.sh`、`orbit358.py`、`sampler358.ps1`
