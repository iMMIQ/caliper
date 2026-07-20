# Caliper

> **精密量具（卡尺）** —— 上传 ONNX，经 ATC 极限优化编译，自研 ACL runner 跑 100 次测平均时延，并用 msprof 取证，全程一个 HTTP API。

面向昇腾（Ascend）NPU 的 ONNX 模型自动化性能表征服务。

> 许可证：**LGPL-3.0**（见 [LICENSE](LICENSE)）

## 流水线

```
上传 ONNX ──▶ ATC 极限编译(→.om) ──▶ warmup ──▶ ACL runner 跑 N 次测时延 ──▶ msprof 取证 ──▶ 返回结果
```

- **编译**：`atc --framework=onnx ... --oo_level=O3 --buffer_optimize=l2_optimize --tiling_schedule_optimize=1`（极限优化预设）
- **基准**：自研 `caliper-runner`，FFI 动态加载 `libascendcl`，warmup 后分别测量模型执行、全部输入 H2D 和全部输出 D2H，统计 mean/p50/p99/min/max/std（μs）
- **取证**：`msprof --application="caliper-runner ..."` 复用同一 runner 采集 profiling 原始数据

## 工作区

| crate | 作用 |
| --- | --- |
| `caliper-core` | 共享类型（JobSpec / Job / 统计结果 / 错误） |
| `caliper-runner` | ACL FFI + 一次性基准二进制（也可被 msprof 包裹） |
| `caliper` | axum 服务：CANN 自动发现、任务编排、多设备独占调度、API |

## 构建

```bash
cargo build --release
# 产物：target/release/caliper、caliper-runner 与 caliper-transfer
```

## H2D / D2H 传输时延实验

`caliper-transfer` 单独测量 host 与 device 之间的同步传输，不需要 ONNX/OM。实验在计时前通过
`aclrtMallocHost` 和 `aclrtMalloc` 分配缓冲区并触碰 host 内存页；预热后，每个样本只包围一次
`aclrtMemcpy`。因此结果表示应用侧可见的单次同步拷贝时延，不包含内存分配、初始化和释放时间。
原有 `caliper-runner` 直接分配并初始化 device buffer，每次迭代只执行 `aclmdlExecute`，其模型
时延统计不包含 H2D/D2H。

```bash
cargo build --release -p caliper-runner --bin caliper-transfer

./target/release/caliper-transfer \
  --lib /usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so \
  --device 0 --iters 100 --warmup 10 \
  --sizes 4K,64K,1M,16M,64M
```

程序向 stdout 输出 JSON。每种大小分别包含 H2D/D2H 的 mean、p50、p99、min、max、stddev
（单位均为 us），以及按平均时延计算的有效带宽（十进制 GB/s）。小消息主要反映固定调用与传输
开销，大消息更适合观察链路吞吐。

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
curl -OJ http://127.0.0.1:7878/v1/jobs/<job_id>/artifacts/atc-pbtxt.tar.gz
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/jobs` | multipart：`spec`(JSON) + `onnx`(文件)，返回 `job_id` |
| `GET` | `/v1/jobs/{id}` | 任务状态与结果 |
| `GET` | `/v1/jobs/{id}/events` | SSE 进度流（见下） |
| `GET` | `/v1/jobs` | 任务列表 |
| `GET` | `/v1/jobs/{id}/artifacts` | 产物清单 |
| `GET` | `/v1/jobs/{id}/artifacts/{name}` | 下载产物（`model.om`/`atc.log`/`atc-pbtxt.tar.gz`/`bench.json`/`msprof.tar.gz`/`result.json`） |
| `DELETE` | `/v1/jobs/{id}` | 取消并清理 |
| `GET` | `/healthz` | 健康检查 |

任务完成后，`GET /v1/jobs/{id}` 的 `result.benchmark.transfer` 返回这个模型一次请求的传输结果：

```json
{
  "input_bytes": 602112,
  "output_bytes": 4000,
  "h2d_latency_us": {
    "mean": 62.1,
    "p50": 61.8,
    "p99": 68.4,
    "min": 60.9,
    "max": 70.2,
    "stddev": 1.6
  },
  "d2h_latency_us": {
    "mean": 11.7,
    "p50": 11.5,
    "p99": 15.9,
    "min": 11.2,
    "max": 17.1,
    "stddev": 0.8
  },
  "h2d_effective_bandwidth_gbps": 9.69,
  "d2h_effective_bandwidth_gbps": 0.34
}
```

H2D 的单个样本覆盖该模型所有输入 tensor 的顺序拷贝，D2H 同理覆盖所有输出 tensor；buffer
分配和初始化不计时。`iterations` 和 `warmup` 与任务的同名字段一致。

## JobSpec 字段

```json
{
  "soc_version": "Ascend310P3",   // 可选，留空自动推断
  "input_shape": "input:1,3,224,224", // 可选，动态形状模型需提供
  "iters": 100,
  "warmup": 10,
  "device_id": null,            // 可选；null 自动选择空闲卡，整数则等待指定卡
  "msprof_iters": 10,
  "extra_atc_flags": "",          // 可选，附加 atc 参数
  "no_cache": false               // 可选，true 则强制重新 ATC 编译、不读不写缓存
}
```

## 多卡独占调度

任务提交后会遍历允许的设备池，对每张卡执行两层检查：

1. 对目标机上的设备锁文件获取非阻塞 `flock`。同机的多个 Caliper 进程、多个用户只要使用同一 `lock_dir`，就不会拿到同一张卡。
2. 持锁后检查 `npu-smi info` 的进程表。卡上已有未遵守 Caliper 锁协议的进程时拒绝调度；无法识别输出时默认 fail closed。

租约覆盖 ATC、benchmark 和 msprof 的完整任务生命周期。任务正常结束、失败或服务进程退出时，内核随文件描述符关闭自动释放租约。没有空闲卡的任务保持 `queued`，`stage` 会给出等待原因；`assigned_device_id` 在拿到卡后记录实际卡号。显式提交 `device_id` 时只等待该卡，省略或设为 `null` 时轮转选择任意空闲卡。

```toml
[devices]
ids = [0, 1, 2, 3]                 # 留空则自动发现
lock_dir = "/run/lock/caliper"     # 所有 Caliper 实例必须完全一致
poll_interval_ms = 1000
require_idle = true
```

### 多人机器的强隔离边界

`flock` 是协作式租约，`npu-smi` 是提交时检查。只要普通用户仍可直接打开 `/dev/davinci*`，任何用户都能绕过调度器并在测量中途启动进程，应用层无法给出严格独占保证。要求性能结果可信时，应把 Caliper 部署成目标机上的单一服务账号：

- 只有 Caliper 服务账号属于有权访问 Ascend 设备节点的用户组，其他用户只调用 HTTP API。
- 使用管理员预建、不可由普通用户删除的 `lock_dir`；若还运行多个 Caliper 实例，则预建每张卡的 `device-<id>.lock` 并授予这些实例共同的组读写权限。
- 保持 `require_idle = true`，用于发现服务启动前已经存在的外部任务或配置错误。

这时权限层阻止绕过，文件租约负责多个 Caliper 实例之间的互斥，二者共同提供严格的一卡一任务约束。容器部署也应只把调度器分配的单个设备节点映射进任务容器，不能把全部 `/dev/davinci*` 暴露给任意用户容器。

## 编译缓存

ATC 编译按 `sha256(onnx) + soc_version + input_shape + extra_atc_flags` 缓存到 `storage/cache/<key>/`。每次编译都会开启 GE 图导出，并将生成的 `.pbtxt` 打包为 `atc-pbtxt.tar.gz`。相同输入的二次提交会同时复用 OM 和 pbtxt 归档，跳过 ATC：

- `JobResult.compile.cached` 标识是否命中（命中时 `duration_ms = 0`）
- 缓存文件通过临时文件原子发布，多卡并发编译不会暴露半写入的 OM 或 pbtxt 归档
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
