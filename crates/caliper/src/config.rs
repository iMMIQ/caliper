//! 配置：toml 文件为基线，命令行参数（Option）覆盖，最后回退到代码默认值。
//! CANN 工具链相关项默认留空 → 运行时自动发现。

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "caliper",
    version,
    about = "Ascend ONNX model characterization service and CLI"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<CliCommand>,
    /// Configuration file path
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,
    /// Server listen address, for example 0.0.0.0:7878
    #[arg(long, global = true)]
    pub bind: Option<String>,
    /// Maximum multipart upload size in MiB
    #[arg(long = "max-upload-mib", global = true)]
    pub max_upload_mib: Option<usize>,
    /// Job storage directory
    #[arg(long, global = true)]
    pub storage: Option<PathBuf>,
    /// Restrict scheduling to one device
    #[arg(long, global = true)]
    pub device: Option<i32>,
    /// Comma-separated device IDs; auto-discover when omitted
    #[arg(long, value_delimiter = ',', global = true)]
    pub devices: Option<Vec<i32>>,
    /// Default benchmark iteration count
    #[arg(long, global = true)]
    pub iters: Option<u32>,
    /// Default warmup iteration count
    #[arg(long, global = true)]
    pub warmup: Option<u32>,
    /// Inference count captured by msprof
    #[arg(long = "msprof-iters", global = true)]
    pub msprof_iters: Option<u32>,
    /// CANN toolkit root (overrides auto-discovery)
    #[arg(long = "cann-home", global = true)]
    pub cann_home: Option<String>,
    /// Target SoC (overrides npu-smi detection)
    #[arg(long = "soc-version", global = true)]
    pub soc_version: Option<String>,
    /// Path to the caliper-runner executable
    #[arg(long, global = true)]
    pub runner: Option<PathBuf>,
    /// Path to libascendcl.so (overrides auto-discovery)
    #[arg(long, global = true)]
    pub libascendcl: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum CliCommand {
    /// Run one ONNX model synchronously and print the result
    Run(RunArgs),
}

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Input ONNX file
    pub onnx: PathBuf,
    /// Dynamic input shape, for example input:1,3,224,224
    #[arg(long = "input-shape")]
    pub input_shape: Option<String>,
    /// Additional ATC arguments
    #[arg(long = "extra-atc-flags")]
    pub extra_atc_flags: Option<String>,
    /// Force compilation without reading or writing the compile cache
    #[arg(long)]
    pub no_cache: bool,
    /// Output the complete job result as JSON (default: human-readable report)
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ConfigFile {
    #[serde(default)]
    pub server: ServerCfg,
    #[serde(default)]
    pub storage: StorageCfg,
    #[serde(default)]
    pub run: RunCfg,
    #[serde(default)]
    pub cann: CannCfg,
    #[serde(default)]
    pub devices: DevicesCfg,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerCfg {
    #[serde(default = "default_bind")]
    pub bind: String,
    #[serde(default = "default_max_upload_mib")]
    pub max_upload_mib: usize,
}
impl Default for ServerCfg {
    fn default() -> Self {
        Self {
            bind: default_bind(),
            max_upload_mib: default_max_upload_mib(),
        }
    }
}
fn default_bind() -> String {
    "0.0.0.0:7878".into()
}
fn default_max_upload_mib() -> usize {
    10 * 1024
}

#[derive(Debug, Clone, Deserialize)]
pub struct StorageCfg {
    #[serde(default = "default_storage")]
    pub dir: PathBuf,
}
impl Default for StorageCfg {
    fn default() -> Self {
        Self {
            dir: default_storage(),
        }
    }
}
fn default_storage() -> PathBuf {
    PathBuf::from("storage")
}

#[derive(Debug, Clone, Deserialize)]
pub struct RunCfg {
    /// 旧配置兼容：设置后等价于只允许这一张卡。
    #[serde(default)]
    pub device_id: Option<i32>,
    #[serde(default = "default_iters")]
    pub iters: u32,
    #[serde(default = "default_warmup")]
    pub warmup: u32,
    #[serde(default = "default_msprof")]
    pub msprof_iters: u32,
}
impl Default for RunCfg {
    fn default() -> Self {
        Self {
            device_id: None,
            iters: default_iters(),
            warmup: default_warmup(),
            msprof_iters: default_msprof(),
        }
    }
}
fn default_iters() -> u32 {
    100
}
fn default_warmup() -> u32 {
    10
}
fn default_msprof() -> u32 {
    10
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct CannCfg {
    #[serde(default)]
    pub home: Option<String>,
    #[serde(default)]
    pub soc_version: Option<String>,
    #[serde(default)]
    pub runner: Option<String>,
    #[serde(default)]
    pub libascendcl: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DevicesCfg {
    /// 空数组表示从 npu-smi info -m 的 Chip Logic ID / /dev/davinci* 自动发现。
    #[serde(default)]
    pub ids: Vec<i32>,
    #[serde(default = "default_device_lock_dir")]
    pub lock_dir: PathBuf,
    #[serde(default = "default_device_poll_interval_ms")]
    pub poll_interval_ms: u64,
    /// true 时 npu-smi 无法确认设备空闲就不调度（fail closed）。
    #[serde(default = "default_true")]
    pub require_idle: bool,
}

impl Default for DevicesCfg {
    fn default() -> Self {
        Self {
            ids: Vec::new(),
            lock_dir: default_device_lock_dir(),
            poll_interval_ms: default_device_poll_interval_ms(),
            require_idle: true,
        }
    }
}

fn default_device_lock_dir() -> PathBuf {
    PathBuf::from("/tmp/caliper-device-locks")
}
fn default_device_poll_interval_ms() -> u64 {
    1000
}
fn default_true() -> bool {
    true
}

/// 解析后的运行时配置。
#[derive(Debug, Clone)]
#[allow(dead_code)] // iters/warmup/msprof_iters 当前由 JobSpec 驱动，留作服务端默认值的预留开关
pub struct Config {
    pub bind: String,
    pub max_upload_bytes: usize,
    pub storage: PathBuf,
    pub device_ids: Vec<i32>,
    pub device_lock_dir: PathBuf,
    pub device_poll_interval_ms: u64,
    pub require_idle_device: bool,
    pub iters: u32,
    pub warmup: u32,
    pub msprof_iters: u32,
    pub cann_home: Option<String>,
    pub soc_version: Option<String>,
    pub runner: Option<PathBuf>,
    pub libascendcl: Option<PathBuf>,
    pub config_path: PathBuf,
}

impl Config {
    pub fn resolve(cli: &Cli) -> Result<Self> {
        let config_path = cli
            .config
            .clone()
            .unwrap_or_else(|| PathBuf::from("config/default.toml"));
        let file: ConfigFile = if config_path.exists() {
            let raw = std::fs::read_to_string(&config_path)
                .with_context(|| format!("读取配置失败: {}", config_path.display()))?;
            toml::from_str(&raw)
                .with_context(|| format!("解析配置失败: {}", config_path.display()))?
        } else {
            ConfigFile::default()
        };

        let max_upload_mib = cli.max_upload_mib.unwrap_or(file.server.max_upload_mib);
        anyhow::ensure!(max_upload_mib > 0, "max_upload_mib 必须大于 0");
        let max_upload_bytes = max_upload_mib
            .checked_mul(1024 * 1024)
            .context("max_upload_mib 数值过大")?;

        Ok(Self {
            bind: cli
                .bind
                .clone()
                .or(Some(file.server.bind))
                .unwrap_or_else(default_bind),
            max_upload_bytes,
            storage: cli
                .storage
                .clone()
                .or(Some(file.storage.dir))
                .unwrap_or_else(default_storage),
            device_ids: cli
                .devices
                .clone()
                .or_else(|| cli.device.map(|d| vec![d]))
                .or_else(|| (!file.devices.ids.is_empty()).then(|| file.devices.ids.clone()))
                .or_else(|| file.run.device_id.map(|d| vec![d]))
                .unwrap_or_default(),
            device_lock_dir: file.devices.lock_dir,
            device_poll_interval_ms: file.devices.poll_interval_ms.max(100),
            require_idle_device: file.devices.require_idle,
            iters: cli.iters.unwrap_or(file.run.iters),
            warmup: cli.warmup.unwrap_or(file.run.warmup),
            msprof_iters: cli.msprof_iters.unwrap_or(file.run.msprof_iters),
            cann_home: cli.cann_home.clone().or(file.cann.home),
            soc_version: cli.soc_version.clone().or(file.cann.soc_version),
            runner: cli
                .runner
                .clone()
                .or_else(|| file.cann.runner.as_ref().map(PathBuf::from)),
            libascendcl: cli
                .libascendcl
                .clone()
                .or_else(|| file.cann.libascendcl.as_ref().map(PathBuf::from)),
            config_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_ten_gib_upload_limit() {
        assert_eq!(
            ServerCfg::default().max_upload_mib,
            10 * 1024,
            "默认配置应允许 10 GiB multipart 请求"
        );
    }

    #[test]
    fn cli_overrides_upload_limit() {
        let missing_config = format!(
            "/tmp/caliper-missing-config-{}-{}.toml",
            std::process::id(),
            uuid::Uuid::new_v4()
        );
        let cli = Cli::try_parse_from([
            "caliper",
            "--config",
            missing_config.as_str(),
            "--max-upload-mib",
            "123",
        ])
        .unwrap();

        let config = Config::resolve(&cli).unwrap();
        assert_eq!(config.max_upload_bytes, 123 * 1024 * 1024);
    }

    #[test]
    fn parses_run_subcommand_and_global_run_options() {
        let cli = Cli::try_parse_from([
            "caliper",
            "run",
            "model.onnx",
            "--iters",
            "7",
            "--device",
            "2",
            "--input-shape",
            "images:1,3,224,224",
            "--no-cache",
        ])
        .unwrap();

        assert_eq!(cli.iters, Some(7));
        assert_eq!(cli.device, Some(2));
        let Some(CliCommand::Run(run)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert_eq!(run.onnx, PathBuf::from("model.onnx"));
        assert_eq!(run.input_shape.as_deref(), Some("images:1,3,224,224"));
        assert!(run.no_cache);
        assert!(!run.json);
    }

    #[test]
    fn parses_explicit_json_output() {
        let cli = Cli::try_parse_from(["caliper", "run", "model.onnx", "--json"]).unwrap();
        let Some(CliCommand::Run(run)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert!(run.json);
    }
}
