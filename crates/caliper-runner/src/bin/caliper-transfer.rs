//! Ascend host/device 同步传输时延微基准。

use anyhow::{Context, Result};
use caliper_core::{stats_from_ns, LatencyStats};
use caliper_runner::acl::Acl;
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "caliper-transfer",
    version,
    about = "Ascend H2D/D2H 同步传输时延基准"
)]
struct Cli {
    /// libascendcl.so 绝对路径
    #[arg(long)]
    lib: PathBuf,
    /// 设备 ID
    #[arg(long, default_value_t = 0)]
    device: i32,
    /// 每个方向、每种大小的计时迭代次数
    #[arg(long, default_value_t = 100)]
    iters: u32,
    /// 每个方向、每种大小的预热迭代次数
    #[arg(long, default_value_t = 10)]
    warmup: u32,
    /// 逗号分隔的传输大小，支持 B/K/KB/KiB/M/MB/MiB/G/GB/GiB
    #[arg(
        long,
        value_delimiter = ',',
        value_parser = parse_size,
        default_value = "4K,64K,1M,16M,64M"
    )]
    sizes: Vec<usize>,
}

#[derive(Serialize)]
struct TransferResult {
    device: i32,
    iterations: u32,
    warmup: u32,
    timer: &'static str,
    copy_api: &'static str,
    host_memory: &'static str,
    allocation_in_timing: bool,
    measurements: Vec<TransferMeasurement>,
}

#[derive(Serialize)]
struct TransferMeasurement {
    size_bytes: usize,
    h2d_latency_us: LatencyStats,
    d2h_latency_us: LatencyStats,
    h2d_effective_bandwidth_gbps: f64,
    d2h_effective_bandwidth_gbps: f64,
}

fn parse_size(raw: &str) -> std::result::Result<usize, String> {
    let normalized: String = raw
        .chars()
        .filter(|c| !c.is_ascii_whitespace())
        .collect::<String>()
        .to_ascii_uppercase();
    let suffixes = [
        ("GIB", 1usize << 30),
        ("GB", 1_000_000_000),
        ("G", 1usize << 30),
        ("MIB", 1usize << 20),
        ("MB", 1_000_000),
        ("M", 1usize << 20),
        ("KIB", 1usize << 10),
        ("KB", 1_000),
        ("K", 1usize << 10),
        ("B", 1),
    ];
    let (digits, multiplier) = suffixes
        .iter()
        .find_map(|(suffix, multiplier)| {
            normalized
                .strip_suffix(suffix)
                .map(|digits| (digits, *multiplier))
        })
        .unwrap_or((&normalized, 1));
    let value = digits
        .parse::<usize>()
        .map_err(|_| format!("无效传输大小: {raw}"))?;
    let bytes = value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("传输大小溢出: {raw}"))?;
    if bytes == 0 {
        return Err("传输大小必须大于 0".into());
    }
    Ok(bytes)
}

fn effective_gbps(size_bytes: usize, latency: &LatencyStats) -> f64 {
    if latency.mean <= 0.0 {
        return 0.0;
    }
    size_bytes as f64 / latency.mean / 1000.0
}

fn run(cli: &Cli) -> Result<TransferResult> {
    if cli.iters == 0 {
        anyhow::bail!("--iters 必须大于 0");
    }
    let mut acl = Acl::open(&cli.lib).context("打开 libascendcl 失败")?;
    acl.init().context("aclInit 失败")?;
    acl.set_device(cli.device)
        .with_context(|| format!("aclrtSetDevice({}) 失败", cli.device))?;

    let mut measurements = Vec::with_capacity(cli.sizes.len());
    for &size in &cli.sizes {
        let (h2d_ns, d2h_ns) = acl
            .measure_transfer_ns(size, cli.iters, cli.warmup)
            .with_context(|| format!("测量 {} bytes 失败", size))?;
        let h2d_latency_us = stats_from_ns(&h2d_ns);
        let d2h_latency_us = stats_from_ns(&d2h_ns);
        measurements.push(TransferMeasurement {
            size_bytes: size,
            h2d_effective_bandwidth_gbps: effective_gbps(size, &h2d_latency_us),
            d2h_effective_bandwidth_gbps: effective_gbps(size, &d2h_latency_us),
            h2d_latency_us,
            d2h_latency_us,
        });
    }

    acl.shutdown();
    Ok(TransferResult {
        device: cli.device,
        iterations: cli.iters,
        warmup: cli.warmup,
        timer: "std::time::Instant",
        copy_api: "synchronous_aclrtMemcpy",
        host_memory: "aclrtMallocHost",
        allocation_in_timing: false,
        measurements,
    })
}

fn main() {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(result) => println!("{}", serde_json::to_string(&result).expect("序列化结果")),
        Err(error) => {
            eprintln!("caliper-transfer failed: {error:#}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_binary_and_decimal_sizes() {
        assert_eq!(parse_size("4K").unwrap(), 4096);
        assert_eq!(parse_size("2 MiB").unwrap(), 2 * 1024 * 1024);
        assert_eq!(parse_size("3MB").unwrap(), 3_000_000);
        assert_eq!(parse_size("17").unwrap(), 17);
    }

    #[test]
    fn rejects_zero_and_invalid_sizes() {
        assert!(parse_size("0").is_err());
        assert!(parse_size("abc").is_err());
    }
}
