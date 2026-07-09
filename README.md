# Caliper

> **精密量具（卡尺）** —— 上传 ONNX，经 ATC 极限优化编译，自研 ACL runner 跑 100 次测平均时延，并用 msprof 取证，全程一个 HTTP API。

面向昇腾（Ascend）NPU 的 ONNX 模型自动化性能表征服务。

> 许可证：**LGPL-3.0**（见 [LICENSE](LICENSE)）

## 流水线

```
上传 ONNX ──▶ ATC 极限编译(→.om) ──▶ warmup ──▶ ACL runner 跑 N 次测时延 ──▶ msprof 取证 ──▶ 返回结果
```

- **编译**：`atc --framework=onnx ... --oo_level=O3 --buffer_optimize=l2_optimize --tiling_schedule_optimize=1`（极限优化预设）
- **基准**：自研 `caliper-runner`，FFI 动态加载 `libascendcl`，warmup 后同步执行 N 次，统计 mean/p50/p99/min/max/std（μs）
- **取证**：`msprof --application="caliper-runner ..."` 复用同一 runner 采集 profiling 原始数据

## 工作区

| crate | 作用 |
| --- | --- |
| `caliper-core` | 共享类型（JobSpec / Job / 统计结果 / 错误） |
| `caliper-runner` | ACL FFI + 一次性基准二进制（也可被 msprof 包裹） |
| `caliper` | axum 服务：CANN 自动发现、任务编排、设备串行、API |

## 构建

```bash
cargo build --release
# 产物：target/release/caliper 与 target/release/caliper-runner
```

## 运行

```bash
# 启动服务（CANN 自动发现，soc 版本从 npu-smi 推断）
./target/release/caliper

# 上传 ONNX
curl -F 'spec={"iters":100,"warmup":10};type=application/json' \
     -F 'onnx=@model.onnx' \
     http://127.0.0.1:7878/v1/jobs

# 轮询状态 / 取结果
curl http://127.0.0.1:7878/v1/jobs/<job_id>
curl -OJ http://127.0.0.1:7878/v1/jobs/<job_id>/artifacts/msprof.tar.gz
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/jobs` | multipart：`spec`(JSON) + `onnx`(文件)，返回 `job_id` |
| `GET` | `/v1/jobs/{id}` | 任务状态与结果 |
| `GET` | `/v1/jobs/{id}/events` | SSE 进度流（见下） |
| `GET` | `/v1/jobs` | 任务列表 |
| `GET` | `/v1/jobs/{id}/artifacts` | 产物清单 |
| `GET` | `/v1/jobs/{id}/artifacts/{name}` | 下载产物（`model.om`/`atc.log`/`bench.json`/`msprof.tar.gz`/`result.json`） |
| `DELETE` | `/v1/jobs/{id}` | 取消并清理 |
| `GET` | `/healthz` | 健康检查 |

## JobSpec 字段

```json
{
  "soc_version": "Ascend310P3",   // 可选，留空自动推断
  "input_shape": "input:1,3,224,224", // 可选，动态形状模型需提供
  "iters": 100,
  "warmup": 10,
  "device_id": 0,
  "msprof_iters": 10,
  "extra_atc_flags": ""           // 可选，附加 atc 参数
  "no_cache": false               // 可选，true 则强制重新 ATC 编译、不读不写缓存
}
```

## 编译缓存

ATC 编译按 `sha256(onnx) + soc_version + input_shape + extra_atc_flags` 缓存到 `storage/cache/<key>/`。相同输入的二次提交直接复用 OM，跳过 ATC：

- `JobResult.compile.cached` 标识是否命中（命中时 `duration_ms = 0`）
- 设备串行锁保证同一时刻只有一个 job 在编译，无并发写竞争
- `spec.no_cache = true` 可强制重编；删除 `storage/cache/` 即清空

## SSE 进度

`GET /v1/jobs/{id}/events` 以 Server-Sent Events 推送进度：

```text
event: progress
data: {"status":"benchmarking","stage":"caliper-runner: 基准中","updated_at":"..."}

event: done            # 进入终态时，推送完整 Job（含 result）
data: {"id":"...","status":"succeeded","result":{...}}

event: error           # 任务不存在或流超时（上限约 1 小时）
```

```bash
curl -N http://127.0.0.1:7878/v1/jobs/<job_id>/events
```
