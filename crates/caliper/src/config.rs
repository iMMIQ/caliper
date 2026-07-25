//! 配置：toml 文件为基线，命令行参数（Option）覆盖，最后回退到代码默认值。
//! CANN 工具链相关项默认留空 → 运行时自动发现。

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "caliper", version, about = "Ascend ONNX 模型性能表征服务")]
pub struct Cli {
    /// 配置文件路径
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// 监听地址，如 0.0.0.0:7878
    #[arg(long)]
    pub bind: Option<String>,
    /// multipart 上传请求上限（MiB）
    #[arg(long = "max-upload-mib")]
    pub max_upload_mib: Option<usize>,
    /// 任务存储目录
    #[arg(long)]
    pub storage: Option<PathBuf>,
    /// 旧版兼容：只允许调度这一张设备
    #[arg(long)]
    pub device: Option<i32>,
    /// 允许调度的设备 ID，逗号分隔；留空自动发现
    #[arg(long, value_delimiter = ',')]
    pub devices: Option<Vec<i32>>,
    /// 默认迭代次数
    #[arg(long)]
    pub iters: Option<u32>,
    /// 默认预热次数
    #[arg(long)]
    pub warmup: Option<u32>,
    /// msprof 采样推理次数
    #[arg(long = "msprof-iters")]
    pub msprof_iters: Option<u32>,
    /// CANN 工具链根目录（覆盖自动发现）
    #[arg(long = "cann-home")]
    pub cann_home: Option<String>,
    /// 目标 SoC（覆盖 npu-smi 推断）
    #[arg(long = "soc-version")]
    pub soc_version: Option<String>,
    /// caliper-runner 可执行文件路径
    #[arg(long)]
    pub runner: Option<PathBuf>,
    /// libascendcl.so 路径（覆盖自动发现）
    #[arg(long)]
    pub libascendcl: Option<PathBuf>,
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
}
