//! Caliper 共享类型：任务规格、状态机、基准/编译/取证结果。
//!
//! 被 `caliper`（服务）和 `caliper-runner`（基准二进制）共同引用，
//! 保证两端序列化结构一致。

use serde::{Deserialize, Serialize};

pub type JobId = String;

/// 单次时延统计（单位：微秒 μs）。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LatencyStats {
    pub mean: f64,
    pub p50: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
}

/// 由 ns 采样序列计算 μs 统计。空切片返回全零。
pub fn stats_from_ns(samples_ns: &[f64]) -> LatencyStats {
    if samples_ns.is_empty() {
        return LatencyStats::default();
    }
    let mut v = samples_ns.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let pct = |p: f64| -> f64 {
        // nearest-rank 百分位
        let idx = ((n - 1.0) * p).round() as usize;
        v[idx.min(v.len() - 1)]
    };
    let var = v
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let to_us = |x: f64| x / 1000.0;
    LatencyStats {
        mean: to_us(mean),
        p50: to_us(pct(0.50)),
        p99: to_us(pct(0.99)),
        min: to_us(v[0]),
        max: to_us(*v.last().unwrap()),
        stddev: to_us(var.sqrt()),
    }
}

/// 模型单个输入/输出的描述（用于报告与 dummy 数据生成）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoDesc {
    pub index: usize,
    pub size_bytes: u64,
    pub shape: Vec<u64>,
}

/// 基准结果：由 caliper-runner 产出的 JSON。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub iterations: u32,
    pub warmup: u32,
    pub device: i32,
    pub latency_us: LatencyStats,
    pub inputs: Vec<IoDesc>,
    pub outputs: Vec<IoDesc>,
}

/// ATC 编译结果。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompileResult {
    pub duration_ms: u64,
    pub soc_version: String,
    pub om_path: String,
    /// 是否命中编译缓存（命中时跳过 ATC，duration_ms 为 0）。
    #[serde(default)]
    pub cached: bool,
}

/// msprof 取证结果（仅原始产物 + 路径，不做服务端解析）。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProfileResult {
    pub duration_ms: u64,
    pub msprof_dir: String,
    pub msprof_tar_gz: String,
}

/// 产物清单条目。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub name: String,
    pub size_bytes: u64,
    pub kind: ArtifactKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Onnx,
    Om,
    Log,
    Bench,
    Profile,
    Result,
}

/// 单任务最终结果。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JobResult {
    pub compile: CompileResult,
    pub benchmark: Option<BenchmarkResult>,
    pub profile: ProfileResult,
    pub artifacts: Vec<Artifact>,
}

/// 任务状态机。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Compiling,
    Benchmarking,
    Profiling,
    Succeeded,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Compiling => "compiling",
            Self::Benchmarking => "benchmarking",
            Self::Profiling => "profiling",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// 用户提交的任务规格（POST /v1/jobs 的 spec 字段）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSpec {
    /// 目标 SoC，如 Ascend310P3。留空则由服务从 npu-smi 推断。
    #[serde(default)]
    pub soc_version: Option<String>,
    /// 动态形状模型需提供，如 "input:1,3,224,224"；静态形状模型可留空。
    #[serde(default)]
    pub input_shape: Option<String>,
    #[serde(default = "default_iters")]
    pub iters: u32,
    #[serde(default = "default_warmup")]
    pub warmup: u32,
    #[serde(default = "default_device")]
    pub device_id: i32,
    #[serde(default = "default_msprof_iters")]
    pub msprof_iters: u32,
    /// 附加 atc 参数（原样拼接），高级用途。
    #[serde(default)]
    pub extra_atc_flags: Option<String>,
    /// 跳过编译缓存，强制重新 ATC 编译。
    #[serde(default)]
    pub no_cache: bool,
}

fn default_iters() -> u32 {
    100
}
fn default_warmup() -> u32 {
    10
}
fn default_device() -> i32 {
    0
}
fn default_msprof_iters() -> u32 {
    10
}

impl Default for JobSpec {
    fn default() -> Self {
        Self {
            soc_version: None,
            input_shape: None,
            iters: default_iters(),
            warmup: default_warmup(),
            device_id: default_device(),
            msprof_iters: default_msprof_iters(),
            extra_atc_flags: None,
            no_cache: false,
        }
    }
}

/// 服务端持有的任务快照（GET /v1/jobs/{id} 的返回体）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: JobId,
    pub spec: JobSpec,
    pub status: JobStatus,
    /// 当前阶段的人类可读说明（如 "atc: compiling..."）。
    pub stage: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<JobResult>,
    /// 工作目录（绝对路径）。
    pub workdir: String,
    /// 上传的原始 onnx 文件名。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub onnx_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_basic() {
        let ns = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]; // 1..5 μs
        let s = stats_from_ns(&ns);
        assert!((s.mean - 3.0).abs() < 1e-9);
        assert!((s.min - 1.0).abs() < 1e-9);
        assert!((s.max - 5.0).abs() < 1e-9);
        // p50 nearest-rank: idx = (4*0.5).round = 2 -> v[2] = 3000ns -> 3.0μs
        assert!((s.p50 - 3.0).abs() < 1e-9);
        // p99: idx = (4*0.99).round = 4 -> v[4] = 5000ns -> 5.0μs
        assert!((s.p99 - 5.0).abs() < 1e-9);
    }

    #[test]
    fn stats_empty() {
        let s = stats_from_ns(&[]);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn spec_defaults() {
        let s = JobSpec::default();
        assert_eq!(s.iters, 100);
        assert_eq!(s.warmup, 10);
        assert_eq!(s.device_id, 0);
        assert!(s.soc_version.is_none());
    }

    #[test]
    fn spec_from_json_uses_defaults() {
        let s: JobSpec = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(s.iters, 100);
        assert_eq!(s.warmup, 10);
    }
}
