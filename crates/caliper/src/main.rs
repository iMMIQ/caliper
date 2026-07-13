//! Caliper 服务入口：解析配置 → 发现 CANN → 定位 runner → 启动 axum。

mod api;
mod cann;
mod config;
mod device;
mod pipeline;
mod state;
mod store;
mod tools;

use anyhow::Result;
use clap::Parser;
use config::{Cli, Config};
use state::AppState;
use std::path::PathBuf;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "caliper=info,tower_http=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let cfg = Config::resolve(&cli)?;
    let bind = cfg.bind.clone();

    let cann = cann::discover(cfg.cann_home.as_deref())?;
    info!(home = %cann.home.display(), "CANN 已发现");
    info!(atc = %cann.atc.display(), msprof = %cann.msprof.display());
    info!(lib = %cann.libascendcl.display(), acl_include = %cann.acl_include.display(), "libascendcl / acl headers");

    // 定位 caliper-runner：--runner > 同目录同名二进制
    let runner: PathBuf = match &cfg.runner {
        Some(p) => p.clone(),
        None => {
            let exe = std::env::current_exe()?;
            let dir = exe
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));
            dir.join("caliper-runner")
        }
    };
    if !runner.exists() {
        anyhow::bail!(
            "未找到 caliper-runner: {}（用 --runner 指定，或确保它与 caliper 在同一目录）",
            runner.display()
        );
    }
    info!(runner = %runner.display(), "runner 路径");

    // 存储目录转绝对路径
    let storage = if cfg.storage.is_absolute() {
        cfg.storage.clone()
    } else {
        std::env::current_dir()?.join(&cfg.storage)
    };
    std::fs::create_dir_all(storage.join("jobs"))?;
    info!(storage = %storage.display(), "存储目录");

    let devices = device::DeviceManager::new(
        cfg.device_ids.clone(),
        cfg.device_lock_dir.clone(),
        cfg.device_poll_interval_ms,
        cfg.require_idle_device,
    )?;
    info!(devices = ?devices.device_ids(), lock_dir = %cfg.device_lock_dir.display(), "NPU 独占调度器已启用");

    if let Some(s) = cfg.soc_version.clone().or_else(|| {
        devices
            .device_ids()
            .first()
            .and_then(|device| cann::infer_soc(*device))
    }) {
        info!(soc = %s, "目标 SoC（可被 JobSpec 覆盖）");
    } else {
        info!("未能从 npu-smi 推断 SoC，需在 JobSpec 中指定 soc_version");
    }

    let state = AppState::new(cfg, cann, devices, runner, storage);
    let app = api::router(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    info!(bind = %bind, "listening");
    axum::serve(listener, app).await?;
    Ok(())
}
