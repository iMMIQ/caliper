//! caliper-runner：一次性基准二进制。
//!
//! 动态加载 libascendcl → 加载 OM → warmup → 同步执行 N 次（每次主机侧计时）→
//! 打印 BenchmarkResult JSON 到 stdout。服务编排它，msprof 也用它作 --application。
//!
//! 用法：
//!   caliper-runner --om model.om --lib /path/libascendcl.so \
//!                  --device 0 --iters 100 --warmup 10

use anyhow::{Context, Result};
use caliper_core::{stats_from_ns, BenchmarkResult, LatencyStats, ModelTransferResult};
use caliper_runner::acl;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "caliper-runner", version, about = "Ascend OM 单模型时延基准")]
struct Cli {
    /// 编译产物 OM 文件路径
    #[arg(long)]
    om: PathBuf,
    /// libascendcl.so 绝对路径
    #[arg(long)]
    lib: PathBuf,
    /// 设备 ID
    #[arg(long, default_value_t = 0)]
    device: i32,
    /// 计时迭代次数
    #[arg(long, default_value_t = 100)]
    iters: u32,
    /// 预热迭代次数
    #[arg(long, default_value_t = 10)]
    warmup: u32,
    /// 测量模型全部输入 H2D 和全部输出 D2H；服务基准开启，msprof 取证不开启
    #[arg(long)]
    measure_transfer: bool,
}

fn effective_gbps(size_bytes: u64, latency: &LatencyStats) -> f64 {
    if latency.mean <= 0.0 {
        return 0.0;
    }
    size_bytes as f64 / latency.mean / 1000.0
}

fn run(cli: &Cli) -> Result<BenchmarkResult> {
    let mut acl = acl::Acl::open(&cli.lib).context("打开 libascendcl 失败")?;
    acl.init().context("aclInit 失败")?;
    acl.set_device(cli.device)
        .with_context(|| format!("aclrtSetDevice({}) 失败", cli.device))?;
    let (inputs, outputs) = acl
        .load_model(&cli.om)
        .with_context(|| format!("加载 OM 失败: {}", cli.om.display()))?;

    // warmup
    for _ in 0..cli.warmup {
        acl.execute().context("warmup aclmdlExecute 失败")?;
    }

    // measure
    let mut samples_ns = Vec::with_capacity(cli.iters as usize);
    for _ in 0..cli.iters {
        let t0 = Instant::now();
        acl.execute().context("aclmdlExecute 失败")?;
        samples_ns.push(t0.elapsed().as_nanos() as f64);
    }

    let transfer = if cli.measure_transfer {
        let (h2d_ns, d2h_ns) = acl
            .measure_model_transfer_ns(cli.iters, cli.warmup)
            .context("测量模型 H2D/D2H 失败")?;
        let input_bytes = inputs.iter().map(|input| input.size_bytes).sum();
        let output_bytes = outputs.iter().map(|output| output.size_bytes).sum();
        let h2d_latency_us = stats_from_ns(&h2d_ns);
        let d2h_latency_us = stats_from_ns(&d2h_ns);
        Some(ModelTransferResult {
            iterations: cli.iters,
            warmup: cli.warmup,
            input_bytes,
            output_bytes,
            h2d_effective_bandwidth_gbps: effective_gbps(input_bytes, &h2d_latency_us),
            d2h_effective_bandwidth_gbps: effective_gbps(output_bytes, &d2h_latency_us),
            h2d_latency_us,
            d2h_latency_us,
        })
    } else {
        None
    };

    let result = BenchmarkResult {
        iterations: cli.iters,
        warmup: cli.warmup,
        device: cli.device,
        latency_us: stats_from_ns(&samples_ns),
        inputs,
        outputs,
        transfer,
    };

    acl.unload_model();
    acl.shutdown();
    Ok(result)
}

fn main() {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(result) => {
            // 成功：stdout 打印 JSON
            println!("{}", serde_json::to_string(&result).expect("序列化结果"));
            std::process::exit(0);
        }
        Err(e) => {
            // 失败：stderr 打印错误链，非零退出
            eprintln!("caliper-runner failed: {e:#}");
            std::process::exit(1);
        }
    }
}
