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
    /// 任务存储目录
    #[arg(long)]
    pub storage: Option<PathBuf>,
    /// 默认设备 ID
    #[arg(long)]
    pub device: Option<i32>,
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
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerCfg {
    #[serde(default = "default_bind")]
    pub bind: String,
}
impl Default for ServerCfg {
    fn default() -> Self {
        Self {
            bind: default_bind(),
        }
    }
}
fn default_bind() -> String {
    "0.0.0.0:7878".into()
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
    #[serde(default)]
    pub device_id: i32,
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
            device_id: 0,
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

/// 解析后的运行时配置。
#[derive(Debug, Clone)]
#[allow(dead_code)] // iters/warmup/msprof_iters 当前由 JobSpec 驱动，留作服务端默认值的预留开关
pub struct Config {
    pub bind: String,
    pub storage: PathBuf,
    pub device_id: i32,
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

        Ok(Self {
            bind: cli
                .bind
                .clone()
                .or(Some(file.server.bind))
                .unwrap_or_else(default_bind),
            storage: cli
                .storage
                .clone()
                .or(Some(file.storage.dir))
                .unwrap_or_else(default_storage),
            device_id: cli.device.unwrap_or(file.run.device_id),
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
