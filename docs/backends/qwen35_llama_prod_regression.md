# Qwen35-llama 迁移构建 vs 生产冻结 exe 生产线同口径对照回归（task 359，2026-10-14）

**一行状态**：native 6/6 全部 exit 0（P×3＋N×3，ABAB 严格串行，每 native 1 请求）；无 fence/无残留；两臂 argv/env/cwd 唯一差异＝exe 路径；6/6 可见 16-token 序列逐字节相同（同一 SHA `795a93accf98b2f3…`）；预登记判定 prefill **PASS**（Δ=+1.19%）、decode **PASS**（Δ=-1.87%），整体 **PASS（迁移构建不明显劣于生产基线）**；`TPS_true_flush` 两臂均不可得（口径见 §3）。

## 1. 设计与执行事实

- **P 臂**＝生产冻结 exe `D:\work\Strata魔改\results\20261010-qwen35-milestone105\strata-105.exe`（sha256 `0e00af8a11274c62789f47c1b5d21aed811f33adb9b79608ffb7df5525a906b6`，130,165,248 B，mtime 2026-10-10 21:30）。
- **N 臂**＝迁移新构建 `D:\work\StrataUnified\strata\backends\qwen35_llama\build\strata.exe`（sha256 `6062a714c38744a6adc1cb052f3242c304a3ff32d158545933834455544dc9b4`，139,208,192 B，mtime 2026-10-14 16:16）。
- **模型（两臂同读、只读）**：`models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`（13,676,723,168 B，declared `66a3ca888ce13482…`；本轮记录 size/mtime，未重复全量 hash）。
- **prompt（生产协议用例）**：`results/20261010-jev-moderation-3k4k/han3k-risky/prompt.tokens`（3869 ids；现盘 sha256 `803fa7393f1351a04ebb9eacda24318d0cd22875ad9422ef726e670c8c7c1be5`）。
- **argv（两臂逐项相同，cwd=`D:/work/Strata魔改`）**：

```
--qwen35-model models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf --qwen35-pack models --tokens-file results/20261010-jev-moderation-3k4k/han3k-risky/prompt.tokens --max-new 16 --max-context 4096 --qwen35-threads 8 --qwen35-cache-slots 3200 --qwen35-ubatch 4096
```

- **env（两臂逐项相同，仅三项生产 env）**：`STRATA_QWEN35_EXPERT_ORDER=count_desc`、`STRATA_QWEN35_PACKED_IDS=1`、`STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE=1`。
  其余门（STOP_EOG/DECODE_TIMESTAMPS/COMPACT_SLOTS/MTP*/SMALL_T*/SLICE_GU_MULTI）**两臂均未设置**（gates_expected 全为 null，见 `pre-env/pre-*.json`）；清 `STRATA_*`/`GGML_*`/`LLAMA_GRAPH*`/`CUDA_LAUNCH_BLOCKING` 后逐项设置，PATH 前置 `/d/work/Strata魔改/runtime/cuda-12.8.1/bin`，child 去 `STRATA_OPENCODE_OBSERVER_LOG_DIR`。
- **argv 偏差登记（新证据触发的显式登记，两臂同口径）**：task §0 argv 原文未列 `--qwen35-pack`；实测两 exe 谱系在缺 pack 时拒绝运行（N 源码 `qwen35_model.cpp:1697` 空 `pack_dir` 即报错退出 2；P 谱系同源要求），且 milestone105 生产协议（`results/20261010-qwen35-milestone105/MILESTONE-105.md` §1 可运行命令）本就含 `--qwen35-pack models`。故按 §0「与该 exe 兼容的最小写法」补回生产协议原值 `--qwen35-pack models`，两臂一致（P1 config 行 `"pack":"models"` 已核）。
- **单 GPU owner**：`tools/gpu-lock.py`（sha256 `337426d6…`）`run` 全流程，`max_launch=1`、`runtime_limit_seconds=300`、wait≤60 s；6/6 native wait attempts 1、wait 0.0 s；child cleanup `all_exited=true`/`active_processes_zero=true`；marker 均 auto-removed（`retained=false`）；批末 lock 1 B 空、无 strata 进程、VRAM 1103 MiB。预算与模式见 `audit/audit-P1.json` 的 `budget` 段。
- **RAM 门**（pre-run `AvailPhys ≥ 8 GiB`）：P1=17448.5 MB(True)；N1=17132.9 MB(True)；N2=17850.6 MB(True)；P2=17246.1 MB(True)；P3=17826.8 MB(True)；N3=17342.5 MB(True)。
- **时间线（native 级，`batch/run-timeline.txt`）**：

```
2026-10-14 17:07:09
P1 rc=0 at 2026-10-14 17:07:41
2026-10-14 17:07:41
N1 rc=0 at 2026-10-14 17:08:03
2026-10-14 17:08:03
N2 rc=0 at 2026-10-14 17:08:25
2026-10-14 17:08:25
P2 rc=0 at 2026-10-14 17:08:47
2026-10-14 17:08:47
P3 rc=0 at 2026-10-14 17:09:09
2026-10-14 17:09:09
N3 rc=0 at 2026-10-14 17:09:41
```

- 窗口：task.md mtime 2026-10-14T17:01:36.040894+08:00 起，批末 elapsed 9.02 min（<20，无超窗）；顺序严格 P1,N1,N2,P2,P3,N3。
- **每 native 实际请求数＝1**（该谱系无 txn 引擎；6/6 raw 各只有 1 组 `prefill_start`→`summary`；样本数=每臂 native 数=3）。
- 未改动任何源码（研究仓/新仓源码、冻结件、生产对象均只读）；新仓仅新增本报告；harness 参数化 diff 见 `results/opencode-execution/strata-unified-prod-regression-359/adaptive-359.md`。

## 2. 两臂逐 native 原文行（verbatim，出自 `raw/raw-*/child_00.log`）

### P1（P 臂（生产冻结））

READY 行：
```
strata qwen35: bridge ready: 40 layers x 256 experts, 2048 x 512 experts top-8, workers 7, native tier on, prompt experts GPU/MMQ (grouped, resident cache + bounded cold ring)
```
`config`（照录）：
```
{"event":"config","model":"models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf","pack":"models","layers":40,"n_embd":2048,"n_ff":512,"experts":256,"top":8,"cache_slots":3200,"ngl":99,"threads":8,"context":4096,"prompt_tokens":3869,"max_new":16,"qwen35_ubatch":4096,"cuda_backend":""}
```
`first_token`（照录）：
```
{"event":"first_token","id":18,"since_prefill_start_seconds":3.7062582,"since_process_start_seconds":9.1080476}
```
`summary`（照录）：
```
{"event":"summary","exit":0,"prompt_tokens":3869,"output_tokens":16,"decode_evaluations":15,"prefill_seconds":3.7047851,"decode_seconds":0.9433379,"ttft_seconds":3.7062582,"prefill_tokens_per_s":1044.33,"decode_evaluations_per_s":15.901,"strata_jobs":1211936,"strata_seconds":3.59892,"decode_layers":601,"decode_hits":3020,"decode_misses":1788,"prefill_layers":39,"prefill_entries":1207128,"cache_slots":3200,"cache_bytes":3932160000,"cache_mib":3750.0,"cache_payload_mib":1929.2,"cache_resident":1788,"hit_overlap":433,"hit_late":1,"hit_inflight":522,"all_hit_layers":105,"prefill_gpu_layers":39,"prefill_resident":0,"prefill_streamed":9073,"prefill_product_calls":1168,"prefill_stream_mib":9753.09,"prefill_workspace_mib":527.69,"prefill_ring_mib":37.50,"prefill_loan_slots":467,"prefill_loan_mib":547.27,"prefill_owned_mib":0.00,"prefill_loan_restored":0,"prefill_loan_refill_ms":0.0,"prefill_loan_why":"","tier_ms":{"ids":189.1,"launch":71.0,"misses":224.9,"combine":13.0,"admit":317.0},"vram_samples":19,"peak_device_vram_mib":7664.4,"device_total_mib":8187.5,"prefill_process_seconds":3.7008216,"prefill_process_calls":1,"prefill_finish_seconds":0.0021641,"decode_eval_seconds":0.8945346,"decode_eval_spans":15,"decode_eval_per_s":16.7685,"decode_observe_seconds":0.0484356,"decode_observe_rounds":16,"decode_first_vram_seconds":3.5e-06,"decode_first_logits_wait_seconds":3.4e-06,"decode_vram_seconds":0.0010383,"decode_vram_samples":16,"vram_sample_seconds_total":0.0011592,"engine_perf_t_eval_ms_first":0,"engine_perf_t_eval_ms_last":894.208,"engine_perf_t_eval_ms_delta":894.208,"engine_perf_n_eval_first":1,"engine_perf_n_eval_last":15,"engine_perf_n_eval_delta":14,"engine_perf_t_p_eval_ms_first":3702.61,"engine_perf_n_p_eval_first":3869,"shared_overlap_marker_layers":0,"shared_overlap_join_layers":0,"decode_scan_seconds":0.0083851,"decode_scan_rounds":16,"decode_sampler_seconds":0.0254524,"decode_sampler_rounds":16,"decode_topk_seconds":0,"decode_topk_rounds":0,"decode_piece_json_flush_seconds":0.0145934,"decode_piece_json_flush_rounds":16,"decode_piece_seconds":2.17e-05,"decode_piece_rounds":16,"decode_piece_hex_seconds":3.74e-05,"decode_piece_hex_rounds":16,"decode_token_write_seconds":0.0145197,"decode_token_write_rounds":16,"decode_flush_seconds":1.9e-06,"decode_flush_rounds":16}
```

### N1（N 臂（迁移新构建））

READY 行：
```
strata qwen35: bridge ready: 40 layers x 256 experts, 2048 x 512 experts top-8, workers 7, native tier on, prompt experts GPU/MMQ (grouped, resident cache + bounded cold ring)
```
`config`（照录）：
```
{"event":"config","model":"models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf","pack":"models","layers":40,"n_embd":2048,"n_ff":512,"experts":256,"top":8,"cache_slots":3200,"ngl":99,"threads":8,"context":4096,"prompt_tokens":3869,"max_new":16,"qwen35_ubatch":4096,"cuda_backend":""}
```
`first_token`（照录）：
```
{"event":"first_token","id":18,"since_prefill_start_seconds":3.6129014,"since_process_start_seconds":6.5627214}
```
`summary`（照录）：
```
{"event":"summary","exit":0,"prompt_tokens":3869,"output_tokens":16,"decode_evaluations":15,"prefill_seconds":3.6112028,"decode_seconds":0.8430898,"ttft_seconds":3.6129014,"prefill_tokens_per_s":1071.39,"decode_evaluations_per_s":17.7917,"strata_jobs":1211936,"strata_seconds":3.45111,"decode_layers":601,"decode_hits":3020,"decode_misses":1788,"prefill_layers":39,"prefill_entries":1207128,"cache_slots":3200,"cache_bytes":3932160000,"cache_mib":3750.0,"cache_payload_mib":1929.2,"cache_resident":1788,"hit_overlap":434,"hit_late":0,"hit_inflight":487,"all_hit_layers":105,"prefill_gpu_layers":39,"prefill_resident":0,"prefill_streamed":9073,"prefill_product_calls":1168,"prefill_stream_mib":9753.09,"prefill_workspace_mib":527.69,"prefill_ring_mib":37.50,"prefill_loan_slots":467,"prefill_loan_mib":547.27,"prefill_owned_mib":0.00,"prefill_loan_restored":0,"prefill_loan_refill_ms":0.0,"prefill_loan_why":"","tier_ms":{"ids":171.8,"launch":84.2,"misses":151.9,"combine":15.0,"admit":298.7},"vram_samples":19,"peak_device_vram_mib":7615.7,"device_total_mib":8187.5,"prefill_process_seconds":3.6069835,"prefill_process_calls":1,"prefill_finish_seconds":0.0021177,"decode_eval_seconds":0.7986746,"decode_eval_spans":15,"decode_eval_per_s":18.7811,"decode_observe_seconds":0.0439426,"decode_observe_rounds":16,"decode_first_vram_seconds":2.8e-06,"decode_first_logits_wait_seconds":2.6e-06,"decode_vram_seconds":0.0013435,"decode_vram_samples":16,"vram_sample_seconds_total":0.0014628,"engine_perf_t_eval_ms_first":0,"engine_perf_t_eval_ms_last":798.295,"engine_perf_t_eval_ms_delta":798.295,"engine_perf_n_eval_first":1,"engine_perf_n_eval_last":15,"engine_perf_n_eval_delta":14,"engine_perf_t_p_eval_ms_first":3608.67,"engine_perf_n_p_eval_first":3869,"shared_overlap_marker_layers":0,"shared_overlap_join_layers":0,"decode_scan_seconds":0.0070651,"decode_scan_rounds":16,"decode_sampler_seconds":0.0247067,"decode_sampler_rounds":16,"decode_topk_seconds":0,"decode_topk_rounds":0,"decode_piece_json_flush_seconds":0.0121611,"decode_piece_json_flush_rounds":16,"decode_piece_seconds":1.57e-05,"decode_piece_rounds":16,"decode_piece_hex_seconds":3.42e-05,"decode_piece_hex_rounds":16,"decode_token_write_seconds":0.0120907,"decode_token_write_rounds":16,"decode_flush_seconds":1.8e-06,"decode_flush_rounds":16,"mtp_txn":false,"legacy_aggregate_authoritative":true}
```

### N2（N 臂（迁移新构建））

READY 行：
```
strata qwen35: bridge ready: 40 layers x 256 experts, 2048 x 512 experts top-8, workers 7, native tier on, prompt experts GPU/MMQ (grouped, resident cache + bounded cold ring)
```
`config`（照录）：
```
{"event":"config","model":"models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf","pack":"models","layers":40,"n_embd":2048,"n_ff":512,"experts":256,"top":8,"cache_slots":3200,"ngl":99,"threads":8,"context":4096,"prompt_tokens":3869,"max_new":16,"qwen35_ubatch":4096,"cuda_backend":""}
```
`first_token`（照录）：
```
{"event":"first_token","id":18,"since_prefill_start_seconds":3.5416435,"since_process_start_seconds":6.2428643}
```
`summary`（照录）：
```
{"event":"summary","exit":0,"prompt_tokens":3869,"output_tokens":16,"decode_evaluations":15,"prefill_seconds":3.5402408,"decode_seconds":0.7585959,"ttft_seconds":3.5416435,"prefill_tokens_per_s":1092.86,"decode_evaluations_per_s":19.7734,"strata_jobs":1211936,"strata_seconds":3.36664,"decode_layers":601,"decode_hits":3020,"decode_misses":1788,"prefill_layers":39,"prefill_entries":1207128,"cache_slots":3200,"cache_bytes":3932160000,"cache_mib":3750.0,"cache_payload_mib":1929.2,"cache_resident":1788,"hit_overlap":432,"hit_late":2,"hit_inflight":539,"all_hit_layers":105,"prefill_gpu_layers":39,"prefill_resident":0,"prefill_streamed":9073,"prefill_product_calls":1168,"prefill_stream_mib":9753.09,"prefill_workspace_mib":527.69,"prefill_ring_mib":37.50,"prefill_loan_slots":467,"prefill_loan_mib":547.27,"prefill_owned_mib":0.00,"prefill_loan_restored":0,"prefill_loan_refill_ms":0.0,"prefill_loan_why":"","tier_ms":{"ids":184.8,"launch":38.7,"misses":131.7,"combine":8.2,"admit":270.8},"vram_samples":19,"peak_device_vram_mib":7653.4,"device_total_mib":8187.5,"prefill_process_seconds":3.5361894,"prefill_process_calls":1,"prefill_finish_seconds":0.0021381,"decode_eval_seconds":0.7230978,"decode_eval_spans":15,"decode_eval_per_s":20.7441,"decode_observe_seconds":0.0351346,"decode_observe_rounds":16,"decode_first_vram_seconds":2.8e-06,"decode_first_logits_wait_seconds":2.6e-06,"decode_vram_seconds":0.000807,"decode_vram_samples":16,"vram_sample_seconds_total":0.0009336,"engine_perf_t_eval_ms_first":0,"engine_perf_t_eval_ms_last":722.782,"engine_perf_t_eval_ms_delta":722.782,"engine_perf_n_eval_first":1,"engine_perf_n_eval_last":15,"engine_perf_n_eval_delta":14,"engine_perf_t_p_eval_ms_first":3537.89,"engine_perf_n_p_eval_first":3869,"shared_overlap_marker_layers":0,"shared_overlap_join_layers":0,"decode_scan_seconds":0.0073047,"decode_scan_rounds":16,"decode_sampler_seconds":0.0156231,"decode_sampler_rounds":16,"decode_topk_seconds":0,"decode_topk_rounds":0,"decode_piece_json_flush_seconds":0.0121973,"decode_piece_json_flush_rounds":16,"decode_piece_seconds":1.52e-05,"decode_piece_rounds":16,"decode_piece_hex_seconds":3.52e-05,"decode_piece_hex_rounds":16,"decode_token_write_seconds":0.0121253,"decode_token_write_rounds":16,"decode_flush_seconds":2e-06,"decode_flush_rounds":16,"mtp_txn":false,"legacy_aggregate_authoritative":true}
```

### P2（P 臂（生产冻结））

READY 行：
```
strata qwen35: bridge ready: 40 layers x 256 experts, 2048 x 512 experts top-8, workers 7, native tier on, prompt experts GPU/MMQ (grouped, resident cache + bounded cold ring)
```
`config`（照录）：
```
{"event":"config","model":"models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf","pack":"models","layers":40,"n_embd":2048,"n_ff":512,"experts":256,"top":8,"cache_slots":3200,"ngl":99,"threads":8,"context":4096,"prompt_tokens":3869,"max_new":16,"qwen35_ubatch":4096,"cuda_backend":""}
```
`first_token`（照录）：
```
{"event":"first_token","id":18,"since_prefill_start_seconds":3.5646874,"since_process_start_seconds":6.3824387}
```
`summary`（照录）：
```
{"event":"summary","exit":0,"prompt_tokens":3869,"output_tokens":16,"decode_evaluations":15,"prefill_seconds":3.5634767,"decode_seconds":0.7600397,"ttft_seconds":3.5646874,"prefill_tokens_per_s":1085.74,"decode_evaluations_per_s":19.7358,"strata_jobs":1211936,"strata_seconds":3.41253,"decode_layers":601,"decode_hits":3020,"decode_misses":1788,"prefill_layers":39,"prefill_entries":1207128,"cache_slots":3200,"cache_bytes":3932160000,"cache_mib":3750.0,"cache_payload_mib":1929.2,"cache_resident":1788,"hit_overlap":434,"hit_late":0,"hit_inflight":530,"all_hit_layers":105,"prefill_gpu_layers":39,"prefill_resident":0,"prefill_streamed":9073,"prefill_product_calls":1168,"prefill_stream_mib":9753.09,"prefill_workspace_mib":527.69,"prefill_ring_mib":37.50,"prefill_loan_slots":467,"prefill_loan_mib":547.27,"prefill_owned_mib":0.00,"prefill_loan_restored":0,"prefill_loan_refill_ms":0.0,"prefill_loan_why":"","tier_ms":{"ids":175.4,"launch":57.9,"misses":134.8,"combine":10.5,"admit":268.9},"vram_samples":19,"peak_device_vram_mib":7638.4,"device_total_mib":8187.5,"prefill_process_seconds":3.5593465,"prefill_process_calls":1,"prefill_finish_seconds":0.0021559,"decode_eval_seconds":0.7247356,"decode_eval_spans":15,"decode_eval_per_s":20.6972,"decode_observe_seconds":0.0348842,"decode_observe_rounds":16,"decode_first_vram_seconds":2.8e-06,"decode_first_logits_wait_seconds":2.4e-06,"decode_vram_seconds":0.000958,"decode_vram_samples":16,"vram_sample_seconds_total":0.0010824,"engine_perf_t_eval_ms_first":0,"engine_perf_t_eval_ms_last":724.344,"engine_perf_t_eval_ms_delta":724.344,"engine_perf_n_eval_first":1,"engine_perf_n_eval_last":15,"engine_perf_n_eval_delta":14,"engine_perf_t_p_eval_ms_first":3561.09,"engine_perf_n_p_eval_first":3869,"shared_overlap_marker_layers":0,"shared_overlap_join_layers":0,"decode_scan_seconds":0.0070441,"decode_scan_rounds":16,"decode_sampler_seconds":0.0154397,"decode_sampler_rounds":16,"decode_topk_seconds":0,"decode_topk_rounds":0,"decode_piece_json_flush_seconds":0.0123966,"decode_piece_json_flush_rounds":16,"decode_piece_seconds":1.7e-05,"decode_piece_rounds":16,"decode_piece_hex_seconds":3.31e-05,"decode_piece_hex_rounds":16,"decode_token_write_seconds":0.0123324,"decode_token_write_rounds":16,"decode_flush_seconds":1.3e-06,"decode_flush_rounds":16}
```

### P3（P 臂（生产冻结））

READY 行：
```
strata qwen35: bridge ready: 40 layers x 256 experts, 2048 x 512 experts top-8, workers 7, native tier on, prompt experts GPU/MMQ (grouped, resident cache + bounded cold ring)
```
`config`（照录）：
```
{"event":"config","model":"models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf","pack":"models","layers":40,"n_embd":2048,"n_ff":512,"experts":256,"top":8,"cache_slots":3200,"ngl":99,"threads":8,"context":4096,"prompt_tokens":3869,"max_new":16,"qwen35_ubatch":4096,"cuda_backend":""}
```
`first_token`（照录）：
```
{"event":"first_token","id":18,"since_prefill_start_seconds":3.6553213,"since_process_start_seconds":6.356733}
```
`summary`（照录）：
```
{"event":"summary","exit":0,"prompt_tokens":3869,"output_tokens":16,"decode_evaluations":15,"prefill_seconds":3.654102,"decode_seconds":0.785149,"ttft_seconds":3.6553213,"prefill_tokens_per_s":1058.81,"decode_evaluations_per_s":19.1047,"strata_jobs":1211936,"strata_seconds":3.48938,"decode_layers":601,"decode_hits":3020,"decode_misses":1788,"prefill_layers":39,"prefill_entries":1207128,"cache_slots":3200,"cache_bytes":3932160000,"cache_mib":3750.0,"cache_payload_mib":1929.2,"cache_resident":1788,"hit_overlap":434,"hit_late":0,"hit_inflight":534,"all_hit_layers":105,"prefill_gpu_layers":39,"prefill_resident":0,"prefill_streamed":9073,"prefill_product_calls":1168,"prefill_stream_mib":9753.09,"prefill_workspace_mib":527.69,"prefill_ring_mib":37.50,"prefill_loan_slots":467,"prefill_loan_mib":547.27,"prefill_owned_mib":0.00,"prefill_loan_restored":0,"prefill_loan_refill_ms":0.0,"prefill_loan_why":"","tier_ms":{"ids":172.1,"launch":54.7,"misses":145.4,"combine":10.6,"admit":279.2},"vram_samples":19,"peak_device_vram_mib":7630.0,"device_total_mib":8187.5,"prefill_process_seconds":3.6499913,"prefill_process_calls":1,"prefill_finish_seconds":0.002121,"decode_eval_seconds":0.7493753,"decode_eval_spans":15,"decode_eval_per_s":20.0167,"decode_observe_seconds":0.0354085,"decode_observe_rounds":16,"decode_first_vram_seconds":2.6e-06,"decode_first_logits_wait_seconds":2.4e-06,"decode_vram_seconds":0.0010112,"decode_vram_samples":16,"vram_sample_seconds_total":0.0011243,"engine_perf_t_eval_ms_first":0,"engine_perf_t_eval_ms_last":749.088,"engine_perf_t_eval_ms_delta":749.088,"engine_perf_n_eval_first":1,"engine_perf_n_eval_last":15,"engine_perf_n_eval_delta":14,"engine_perf_t_p_eval_ms_first":3651.69,"engine_perf_n_p_eval_first":3869,"shared_overlap_marker_layers":0,"shared_overlap_join_layers":0,"decode_scan_seconds":0.006996,"decode_scan_rounds":16,"decode_sampler_seconds":0.0165032,"decode_sampler_rounds":16,"decode_topk_seconds":0,"decode_topk_rounds":0,"decode_piece_json_flush_seconds":0.011904,"decode_piece_json_flush_rounds":16,"decode_piece_seconds":1.75e-05,"decode_piece_rounds":16,"decode_piece_hex_seconds":3.44e-05,"decode_piece_hex_rounds":16,"decode_token_write_seconds":0.0118354,"decode_token_write_rounds":16,"decode_flush_seconds":1.4e-06,"decode_flush_rounds":16}
```

### N3（N 臂（迁移新构建））

READY 行：
```
strata qwen35: bridge ready: 40 layers x 256 experts, 2048 x 512 experts top-8, workers 7, native tier on, prompt experts GPU/MMQ (grouped, resident cache + bounded cold ring)
```
`config`（照录）：
```
{"event":"config","model":"models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf","pack":"models","layers":40,"n_embd":2048,"n_ff":512,"experts":256,"top":8,"cache_slots":3200,"ngl":99,"threads":8,"context":4096,"prompt_tokens":3869,"max_new":16,"qwen35_ubatch":4096,"cuda_backend":""}
```
`first_token`（照录）：
```
{"event":"first_token","id":18,"since_prefill_start_seconds":4.1159273,"since_process_start_seconds":6.8997707}
```
`summary`（照录）：
```
{"event":"summary","exit":0,"prompt_tokens":3869,"output_tokens":16,"decode_evaluations":15,"prefill_seconds":4.1135898,"decode_seconds":0.800142,"ttft_seconds":4.1159273,"prefill_tokens_per_s":940.541,"decode_evaluations_per_s":18.7467,"strata_jobs":1211936,"strata_seconds":3.8867,"decode_layers":601,"decode_hits":3020,"decode_misses":1788,"prefill_layers":39,"prefill_entries":1207128,"cache_slots":3200,"cache_bytes":3932160000,"cache_mib":3750.0,"cache_payload_mib":1929.2,"cache_resident":1788,"hit_overlap":434,"hit_late":0,"hit_inflight":526,"all_hit_layers":105,"prefill_gpu_layers":39,"prefill_resident":0,"prefill_streamed":9073,"prefill_product_calls":1168,"prefill_stream_mib":9753.09,"prefill_workspace_mib":527.69,"prefill_ring_mib":37.50,"prefill_loan_slots":467,"prefill_loan_mib":547.27,"prefill_owned_mib":0.00,"prefill_loan_restored":0,"prefill_loan_refill_ms":0.0,"prefill_loan_why":"","tier_ms":{"ids":166.6,"launch":75.6,"misses":149.8,"combine":13.8,"admit":274.9},"vram_samples":19,"peak_device_vram_mib":7581.9,"device_total_mib":8187.5,"prefill_process_seconds":4.1097085,"prefill_process_calls":1,"prefill_finish_seconds":0.0018889,"decode_eval_seconds":0.7581747,"decode_eval_spans":15,"decode_eval_per_s":19.7844,"decode_observe_seconds":0.0414964,"decode_observe_rounds":16,"decode_first_vram_seconds":3.9e-06,"decode_first_logits_wait_seconds":3.3e-06,"decode_vram_seconds":0.0010938,"decode_vram_samples":16,"vram_sample_seconds_total":0.0011986,"engine_perf_t_eval_ms_first":0,"engine_perf_t_eval_ms_last":757.811,"engine_perf_t_eval_ms_delta":757.811,"engine_perf_n_eval_first":1,"engine_perf_n_eval_last":15,"engine_perf_n_eval_delta":14,"engine_perf_t_p_eval_ms_first":4111.2,"engine_perf_n_p_eval_first":3869,"shared_overlap_marker_layers":0,"shared_overlap_join_layers":0,"decode_scan_seconds":0.0072129,"decode_scan_rounds":16,"decode_sampler_seconds":0.02176,"decode_sampler_rounds":16,"decode_topk_seconds":0,"decode_topk_rounds":0,"decode_piece_json_flush_seconds":0.012514,"decode_piece_json_flush_rounds":16,"decode_piece_seconds":1.63e-05,"decode_piece_rounds":16,"decode_piece_hex_seconds":3.45e-05,"decode_piece_hex_rounds":16,"decode_token_write_seconds":0.0124421,"decode_token_write_rounds":16,"decode_flush_seconds":2.1e-06,"decode_flush_rounds":16,"mtp_txn":false,"legacy_aggregate_authoritative":true}
```

## 3. 指标口径（显式标注）

- 两臂共同可用字段（均出自各自 `{"event":"summary"}`）：`prompt_tokens`/`output_tokens`/`prefill_seconds`/`prefill_tokens_per_s`/`ttft_seconds`/`decode_seconds`/`decode_evaluations`/`decode_evaluations_per_s`/`decode_hits`/`decode_misses`/`cache_*`/`peak_device_vram_mib`；CLI 可见 token 行 `{"event":"token"}` 与 `first_token`。
- **decode 段 tok/s（本任务可比量）**＝`decode_evaluations_per_s`（= decode_evaluations 15 / decode_seconds），两臂同字段同链；`decode_tps_source` 6/6 均为 `decode_evaluations_per_s`。
- **`TPS_true_flush`/`TPOT_true_flush` 两臂均不可得**：生产 exe 无 `DECODE_TIMESTAMPS` 门（§0 明确不设），逐 token flush 时间戳数组不存在；按预登记「不要用另一臂的字段硬套」，本报告**不给出**该口径的 TPS/ITL 数值。
- `finish=length`（6/6 emitted=16=max_new）为**设计口径**：两臂都不设 STOP_EOG、都「到底生成 max_new」，才逐位可比；本轮**不是** EOS/生产验收形状，也不含任何 EOS 声明。
- 本报告的 decode ~15–20 tok/s 是**无 MTP/SMALL_T/COMPACT 等实验门**的生产协议口径值，**不是产品 TPS**；30–60 档目标数字未引用。

## 4. 对照表与预登记判定

### 4.1 逐 native（P1,N1,N2,P2,P3,N3）

| case | exe | emitted | finish | prefill tok/s | decode tok/s | prefill s | TTFT s | decode s | peak VRAM MiB | exit |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | P | 16 | length | 1044.33 | 15.901 | 3.705 | 3.706 | 0.943 | 7664.4 | 0 |
| N1 | N | 16 | length | 1071.39 | 17.7917 | 3.611 | 3.613 | 0.843 | 7615.7 | 0 |
| N2 | N | 16 | length | 1092.86 | 19.7734 | 3.540 | 3.542 | 0.759 | 7653.4 | 0 |
| P2 | P | 16 | length | 1085.74 | 19.7358 | 3.563 | 3.565 | 0.760 | 7638.4 | 0 |
| P3 | P | 16 | length | 1058.81 | 19.1047 | 3.654 | 3.655 | 0.785 | 7630.0 | 0 |
| N3 | N | 16 | length | 940.541 | 18.7467 | 4.114 | 4.116 | 0.800 | 7581.9 | 0 |

### 4.2 预登记规则与本轮数值

规则（task §3，先于运行登记）：每臂 native 级中位数；Δ=N_median−P_median；漂移带=各臂内极差最大值；**PASS**=两指标都 |Δ| ≤ max(5%×P_median, 漂移带) 且配对无系统性单边（≥2 个时间序配对同向为负才算系统）；**FAIL**=任一指标 Δ<−5% 且 <−漂移带 且配对同向为负。

| 指标 | P 三值(P1,P2,P3) | N 三值(N1,N2,N3) | P_med | N_med | Δ | Δ% | 漂移带 | 阈值 | 配对同负 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1044.33, 1085.74, 1058.81 | 1071.39, 1092.86, 940.54 | 1058.81 | 1071.39 | +12.580000 | +1.1881% | 152.319 | 152.319 | 1/3 | **PASS** |
| decode | 15.90, 19.74, 19.10 | 17.79, 19.77, 18.75 | 19.1047 | 18.7467 | -0.358000 | -1.8739% | 3.8348 | 3.8348 | 1/3 | **PASS** |

- 配对明细（按 ABAB 时间序配对 N_i−P_i）：prefill＝N1−P1=+27.060, N2−P2=+7.120, N3−P3=-118.269；decode＝N1−P1=+1.8907, N2−P2=+0.0376, N3−P3=-0.3580。
- **整体判定：PASS** — both metrics PASS。
- 两指标均无系统性单边（各只有 1 个配对为负）；decode Δ=−1.87% 远小于漂移带（±3.8348），且分位皆在臂内离散范围内。

### 4.3 可见输出序列（如实）

- 6/6 的 `index:id` 序列 SHA256 全等＝`795a93accf98b2f398a9037fa13e7a41363a369c59756d626329101f6439b165`；可见文本 6/6 全等＝`3<|im_end|>
<|endoftext|><|im_start|>assistant
<think>

</think>

3<|im_end|>
<|endoftext|><|im_start|>`。
- ids（16）：`[18, 248046, 198, 248044, 248045, 74455, 198, 248068, 271, 248069, 271, 18, 248046, 198, 248044, 248045]`。
- 说明：两臂为不同源码谱系，本不要求输出相同；本样本下生产冻结与新构建选择了完全一致的 16-token 序列（同一 max_new=16、无采样门差异）。

## 5. 资源与采样（10 s 采样）

| case | 采样数 | AvailPhys 最低 MB | 页文件峰值 MB | 采样 VRAM 峰值 MiB | util 峰值 pct | 引擎 peak VRAM MiB |
|---|---|---|---|---|---|---|
| P1 | 3 | 8967.3 | 19959.9 | 7579.0 | 98.0 | 7664.4 |
| N1 | 2 | 3284.9 | 15548.1 | 3746.0 | 39.0 | 7615.7 |
| N2 | 2 | 7253.2 | 15516.4 | 3746.0 | 15.0 | 7653.4 |
| P2 | 2 | 5258.9 | 15507.0 | 3762.0 | 49.0 | 7638.4 |
| P3 | 2 | 6190.4 | 15543.5 | 3798.0 | 48.0 | 7630.0 |
| N3 | 3 | 2921.1 | 17195.5 | 7537.0 | 39.0 | 7581.9 |

- postfallback（`batch/postfallback-*.txt`）：6/6 无残留 strata 进程；批末 VRAM 1103 MiB、util 1%。
- 引擎静态量 6/6 一致：prompt_tokens=3869、decode_hits=3020、decode_misses=1788、cache_resident=1788、cache_mib=3750.0、cache_slots=3200、strata_jobs=1211936（P1/N1 抽样原文见 §2）。

## 6. 缺口与登记事项

1. `TPS_true_flush`/`TPOT`/逐 token ITL 两臂均不可得（无逐 token 时间戳门）；30–60 TPS 目标、产品速度声明均不适用本报告。
2. n=3/arm 单机单次小样本：按预登记区间判定 PASS，不构成统计显著性推断；漂移带偏大（N3 prefill 4.11 s 为臂内低值），后续如需收紧可另行登记重复。
3. argv 补 `--qwen35-pack models` 的偏差已登记（§1）；其余 argv/env 与 task §0 逐字一致。
4. `job/` 内残留 `job-A*.json`/`job-B*.json`（m355/m358 遗留副本）不参与本轮；本轮实际使用 `job/job-P1..N3.json`（由 `orbit359.py jobs` 生成）。
5. 禁用数字未引用；EXPERIMENTAL（MTP/COMPACT 等）未写成已认证；无失败原件（6/6 exit 0、marker 无保留、无补跑）。

## 7. 证据清单（`results/opencode-execution/strata-unified-prod-regression-359/`）

- `task.md`、`codex359-review-next.md`、`adaptive-359.md`（harness 参数化 diff）、`harness-copy-sha359.txt`、`harness-pristine358/`（358 逐字节副本留档）
- `job/job-P1..N3.json`；`audit/audit-P*.json`/`audit-N*.json`；`result/result-*.json`、`result/*-ready.json`；`raw/raw-*/child_00.log`（原生 stdout 全量）
- `pre-env/pre-*.json`、`pre-env/pre-*-ramgate.json`；`samples/samples-*.csv`＋`samples-*-summary.json`；`parse-*.json`、`compare-359.json`、`stats-359.json`
- `batch/start.json`、`batch/end.json`、`batch/run-timeline.txt`、`batch/timecheck-*.txt`、`batch/postfallback-*.txt`、`batch/*-markercheck.txt`、`batch/*-result-note.txt`、`events.jsonl`、`request-evidence.jsonl`
- 运行器：`orbit359.py`、`run359.sh`、`sampler359.ps1`、`stats359.py`、`gen-report-359.py`（参数化 diff 见 `adaptive-359.md`）
