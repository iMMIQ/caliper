//! caliper-runner：一次性基准二进制。
//!
//! 动态加载 libascendcl → 加载 OM → warmup → 同步执行 N 次（每次主机侧计时）→
//! 打印 BenchmarkResult JSON 到 stdout。服务编排它，msprof 也用它作 --application。
//!
//! 用法：
//!   caliper-runner --om model.om --lib /path/libascendcl.so \
//!                  --device 0 --iters 100 --warmup 10

mod acl;

use anyhow::{Context, Result};
use caliper_core::{stats_from_ns, BenchmarkResult};
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

    let result = BenchmarkResult {
        iterations: cli.iters,
        warmup: cli.warmup,
        device: cli.device,
        latency_us: stats_from_ns(&samples_ns),
        inputs,
        outputs,
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
