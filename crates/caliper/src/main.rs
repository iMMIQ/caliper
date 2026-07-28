//! Caliper 入口：解析配置后启动 HTTP 服务，或同步执行一个 CLI 模型任务。

mod api;
mod cann;
mod config;
mod device;
mod pipeline;
mod report;
mod state;
mod store;
mod tools;

use anyhow::Result;
use caliper_core::{Job, JobSpec, JobStatus};
use clap::Parser;
use config::{Cli, CliCommand, Config, RunArgs};
use state::AppState;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let default_filter = if matches!(&cli.command, Some(CliCommand::Run(_))) {
        "caliper=warn,tower_http=warn"
    } else {
        "caliper=info,tower_http=info"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| default_filter.into()),
        )
        .init();

    let cfg = Config::resolve(&cli)?;
    let requested_device = cli.device;
    let command = cli.command;
    let state = build_state(cfg)?;

    match command {
        Some(CliCommand::Run(args)) => run_once(state, args, requested_device).await,
        None => serve(state).await,
    }
}

fn build_state(cfg: Config) -> Result<Arc<AppState>> {
    let mut cann = cann::discover(cfg.cann_home.as_deref())?;
    if let Some(libascendcl) = &cfg.libascendcl {
        anyhow::ensure!(
            libascendcl.is_file(),
            "libascendcl.so 不存在: {}",
            libascendcl.display()
        );
        cann.libascendcl = libascendcl.clone();
    }
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

    Ok(AppState::new(cfg, cann, devices, runner, storage))
}

async fn serve(state: Arc<AppState>) -> Result<()> {
    let bind = state.cfg.bind.clone();
    info!(
        max_upload_mib = state.cfg.max_upload_bytes / (1024 * 1024),
        "上传限制"
    );
    let app = api::router(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    info!(bind = %bind, "listening");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn run_once(
    state: Arc<AppState>,
    args: RunArgs,
    requested_device: Option<i32>,
) -> Result<()> {
    let json_output = args.json;
    anyhow::ensure!(
        args.onnx.is_file(),
        "ONNX 文件不存在: {}",
        args.onnx.display()
    );
    let source = std::fs::canonicalize(&args.onnx)
        .map_err(anyhow::Error::from)
        .map_err(|error| anyhow::anyhow!("读取 ONNX 路径失败 {}: {error}", args.onnx.display()))?;
    let id = uuid::Uuid::new_v4().to_string();
    let workdir = store::job_dir(&state.storage, &id);
    std::fs::create_dir_all(&workdir)?;
    let uploading = workdir.join("model.onnx.uploading");
    let destination = store::onnx_path(&workdir);
    tokio::fs::copy(&source, &uploading)
        .await
        .map_err(anyhow::Error::from)
        .map_err(|error| anyhow::anyhow!("复制 ONNX 失败 {}: {error}", source.display()))?;
    tokio::fs::rename(&uploading, &destination).await?;

    let sha256 = pipeline::sha256_file(&destination)?;
    let onnx_name = source
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned);
    let spec = JobSpec {
        soc_version: state.cfg.soc_version.clone(),
        input_shape: args.input_shape,
        iters: state.cfg.iters,
        warmup: state.cfg.warmup,
        device_id: requested_device,
        msprof_iters: state.cfg.msprof_iters,
        extra_atc_flags: args.extra_atc_flags,
        no_cache: args.no_cache,
    };
    write_cli_meta(&workdir, &id, &spec, onnx_name.as_deref(), &sha256)?;

    let now = chrono::Utc::now();
    state
        .insert_job(Job {
            id: id.clone(),
            spec,
            status: JobStatus::Queued,
            stage: "排队中".into(),
            created_at: now,
            updated_at: now,
            error: None,
            result: None,
            workdir: workdir.to_string_lossy().into_owned(),
            onnx_name,
            assigned_device_id: None,
        })
        .await;
    state.register_cancel(&id).await;
    info!(job_id = %id, onnx = %source.display(), "CLI 任务已创建");

    pipeline::run_pipeline(state.clone(), id.clone()).await;
    let job = state
        .get_job(&id)
        .await
        .ok_or_else(|| anyhow::anyhow!("CLI 任务执行后丢失: {id}"))?;
    let mut stdout = std::io::stdout().lock();
    report::write_job(&mut stdout, &job, json_output)?;

    if job.status != JobStatus::Succeeded {
        anyhow::bail!(
            "任务 {} {}: {}（现场: {}）",
            job.id,
            job.status.as_str(),
            job.error.as_deref().unwrap_or(&job.stage),
            job.workdir
        );
    }
    Ok(())
}

fn write_cli_meta(
    workdir: &Path,
    id: &str,
    spec: &JobSpec,
    onnx_name: Option<&str>,
    sha256: &str,
) -> Result<()> {
    let meta = serde_json::json!({
        "id": id,
        "spec": spec,
        "onnx_name": onnx_name,
        "sha256": sha256,
        "source": "cli",
    });
    std::fs::write(store::meta_json(workdir), serde_json::to_vec_pretty(&meta)?)?;
    Ok(())
}
