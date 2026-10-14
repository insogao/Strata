# qwen35_llama serve 直连协议冒烟（任务 363）

2026-10-14 19:20–19:24 +08:00（主管重登记 20 min 窗口锚 19:20:38 后执行；前两次会话因窗口/工具问题零 native 或未起测，见任务仓检查点）。单次诊断口径，不作产品速度。

## 0. 被测对象

- exe：`backends/qwen35_llama/build/strata.exe`，sha256 `2f47b88a9969d04ed5a6fafed6a561b80f51c2945d054e0828e9d9c4a297ebd1`（申报=lease audit 实测一致），139,216,384 B（362 构建）。
- 模型（只读）：`models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`（`66a3ca88…`）+ `--qwen35-pack models`。
- argv（两例相同）：`--qwen35-model models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf --qwen35-pack models --serve --max-context 4096 --qwen35-threads 8 --qwen35-cache-slots 3200 --qwen35-ubatch 4096`。
- env（仅生产三项）：`STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE=1`、`STRATA_QWEN35_PACKED_IDS=1`、`STRATA_QWEN35_EXPERT_ORDER=count_desc`；启动前清 `STRATA_*`/`GGML_*`/`LLAMA_GRAPH*`/`CUDA_LAUNCH_BLOCKING`。
- 租约：`tools/gpu-lock.py run`，每例一租约，`max_launch=1`、`runtime_limit_seconds=420`、wait≤60 s；两例 child exit 0，无 fence 保留（`marker_exists: False`）。native 预算 3/4（首例空格版 stdin 失败 1 + 本两例 2）。
- 输入（**逗号分隔为官方客户端契约**，`serve/server.py:1518/:1633` 发送 `','.join(ids)`；362 实现按此契约并拒绝 id 列表内空格，`src/program/qwen35_serve.cpp:174-176`）：
  - `s1-stdin-commas.txt` sha256 `b5d0fcef509be0eb3743ae0f02860ca77a7cd557b3ebf476f5ae489228eff48a`（GEN 16 P1=3869 ids / GEN 16 P2=3902 ids / GEN 16 P1）。
  - `s2-stdin-commas.txt` sha256 `19e311017037bd7572a5160ef31c036b299aaf5d2c6c495eb422e0c06e1958a3`（GEN 512 P1 / STOP / GEN 16 P1）。
- DONE 字段映射按 `serve/server.py::_parse_done`：`DONE <produced> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <accepted> <offered> <reused> <hits> <lookups> <ram_blobs> <file_blobs> <file_mb> <prompt_read> <offloaded>`。

## 1. S1c（连续请求＋隔离）协议原文

`raw/raw-S1c/child_00.log:190-222` 照录：

```
READY 4096 stop
PP 3869 3869 5726 676.3
T 18
T 248046
DONE 2 3869 5728.2 121.4 stop 0 0 0 0 0 0 0 0.0 3869 0
PP 3902 3902 2821 1383.6
T 90
T 198
T 1
T 78151
T 763
T 220
T 17
T 11
T 198
T 1
T 9676
T 33805
T 763
T 804
T 198
T 92
DONE 16 3902 2823.0 907.4 length 0 0 0 0 0 0 0 0.0 3902 0
PP 3869 3869 2702 1432.6
T 18
T 248046
DONE 2 3869 2703.7 33.6 stop 0 0 0 0 0 0 0 0.0 3869 0
```

无 ERR 行。

### 逐请求判定（S1c，lease 19:21:23.673 起，child pid 51196 exit 0，native elapsed 20.266 s）

| # | prompt | produced | finish | prompt_tokens | prompt_ms | decode_ms | reused/hits/lookups | prompt_read | offloaded |
|---|--------|----------|--------|---------------|-----------|-----------|---------------------|-------------|-----------|
| 1 | P1 3869 | 2 | stop | 3869 | 5728.2 | 121.4 | 0/0/0 | 3869 | 0 |
| 2 | P2 3902 | 16 | length | 3902 | 2823.0 | 907.4 | 0/0/0 | 3902 | 0 |
| 3 | P1 3869 | 2 | stop | 3869 | 2703.7 | 33.6 | 0/0/0 | 3869 | 0 |

隔离证据：
- req1 与 req3（同 P1）可见 token 序列**逐字节相同**；两处 T 行原文 `T 18\nT 248046\n`，sha256 `12699c3c20024161ee7793ac61474803921223992f1cdea621ca95b86cd2eed6`（两请求同 SHA）。
- req2（P2）序列不同，前 8 个 id：`90,198,1,78151,763,220,17,11`（16 token 全序列 sha256 `70064546694dbeb4b1508e0d5af280a9d41b3197787fa521ff077ed46d62aadd`）。

### 与预登记期望的偏差（如实）

- 期望「3/3 finish=stop，各 16 token」；实际 req1/req3 = `stop` @ produced=2（temperature=0 下第 2 个 token 即 EOG `248046`，两次确定性一致），req2 = `length` @ 16（预算用尽未遇 EOG）。
- 即：**协议流、EOS(stop)、取消位清除、每请求状态清零（reused/hits/lookups=0，prompt_read=全长）均按协议呈现**；但「P1/P2 必出 16 token 后 EOS」这一 fixture 期望不成立（P1 短 EOS、P2 不 EOS）。属 363 fixture 预登记错误，非引擎缺陷。decode 数字为单次诊断（如 req1 2/0.1214 ≈ 16.5 tok/s、req3 33.6 ms），不作产品速度。

## 2. S2c（取消路径）协议原文

`raw/raw-S2c/child_00.log:190-197` 照录：

```
READY 4096 stop
ERR prompt (3869 tokens) + max_new (512) exceeds the context (4096)
PP 3869 3869 3488 1109.8
T 18
T 248046
DONE 2 3869 3490.1 100.4 stop 0 0 0 0 0 0 0 0.0 3869 0
```

### 逐请求判定（S2c，lease 19:23:04.373 起，child pid 40292 exit 0，native elapsed 7.687 s）

| # | 输入 | 结果 |
|---|------|------|
| 1 | GEN 512 P1 | **ERR**（3869+512>4096，按上下文门拒绝）——无 PP/T/DONE，无 `finish=cancel`，未进入 prefill/decode |
| — | STOP | 落在请求之间；按已文档化语义「请求之间的 STOP 为 stale，下一请求开始时丢弃」（`qwen35_serve.cpp:456-457`、协议文档 `qwen35_llama_serve_protocol.md` STOP 行） |
| 2 | GEN 16 P1 | 正常完成：`stop` @ produced=2，序列 `T 18 / T 248046`（与 S1c req1/req3 逐字节相同，同 SHA `12699c3c…`）→ 进程在 ERR 后存活、下一请求状态正确（串例隔离佐证） |

### 与预登记期望的偏差（如实）

- 期望「req1 finish=cancel」；实际 req1 被**合法上下文检查**拒绝（ERR 原文如上），cancel 语义**未被本次冒烟触达**。原因：fixture 内部矛盾——P1=3869 tokens 与本任务合法 ctx=4096 下，`GEN 512` 不可能获得合法解码预算（合同亦规定长输出须合法更大 context）。
- 「取消后进程存活、下一请求正常」结论成立（ERR→serving→req2 stop），但 **STOP 取消活跃请求**这一点**未验证**，不写成已验证。后续验证需合法请求（更短 prompt 或更大 context）。

## 3. 资源

- S1c：RAM 门 `avail_phys 15637.3 MB` 通过；采样 4 点，`avail_phys_min 2884.0 MB`、页文件峰值 `25256.2 MB`、`vram_used_max 7665 MiB`、`gpu_util_max 99%`（`samples/samples-S1c-summary.json`）。
- S2c：RAM 门 `avail_phys 17115.7 MB` 通过；采样仅 2 点（10 s 间隔），child 仅 7.687 s，**采样窗错过活动段**——报告值 `vram_used_max 1226 MiB` 为启动前/退出时基线，**非峰值**；`avail_phys_min 15992.2 MB`、页文件峰 `16692.9 MB`（`samples/samples-S2c-summary.json`）。
- serve 模式 child 日志无 `summary` 事件（0 次匹配）⇒ expert tier cache hits/misses 计数在本模式**不可得**；DONE 中 `hits/lookups/reused` 全 0（阶段一不跨请求携带任何前缀）。
- postfallback：S1c 19:21:52 `strata_processes: <none>`、VRAM 1228/8188 MiB、GPU 0%；无残留进程。

## 4. 缺口与结论（一句话）

- 通过：READY/PP/T/DONE 行协议连续 3 请求可用；EOS→`finish=stop`、预算→`finish=length` 正确；同 prompt 同结果（req1==req3 逐字节，SHA 同）、异 prompt 异结果；`reused/hits/lookups=0`、`prompt_read=全长`；ERR 非致命、进程存活；无 fence、exit 0。
- 缺口：①cancel（STOP 取消活跃请求）未触达——S2 fixture 的 GEN 512 被 ctx4096 合法拒绝，建议后续任务用合法请求复核；②S2c VRAM 峰值未采到；③「P1/P2 16-token EOS」预登记期望与 fixture 实际行为不符（P1 EOS@2、P2 length@16）。
- 证据：`D:/work/Strata魔改/results/opencode-execution/strata-unified-serve-smoke-363/`（job/audit/result/raw/pre-env/samples/批前后状态；本次新增解析制品 `parse-S1c.json`、`parse-S2c.json`，含逐请求 T 序列与序列 SHA；runner 同族脚本 `orbit363.py`）。
