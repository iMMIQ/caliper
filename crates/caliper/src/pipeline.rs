//! 任务流水线编排：acquire 设备锁 → ATC 极限编译 → caliper-runner 基准 → msprof 取证 → 汇总。
//! 设备相关步骤全程持锁，保证同一设备串行。

use crate::device::{DeviceLease, TryAcquire};
use crate::state::AppState;
use crate::store;
use crate::tools;
use anyhow::{Context, Result};
use caliper_core::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Instant;
use tokio::process::Command;

pub async fn run_pipeline(state: Arc<AppState>, job_id: String) {
    let res = run_inner(state.clone(), &job_id).await;
    match res {
        Ok(()) => {
            let cancelled = state
                .get_job(&job_id)
                .await
                .map(|j| j.status == JobStatus::Cancelled)
                .unwrap_or(false);
            if !cancelled {
                state
                    .update_job(&job_id, |j| {
                        j.status = JobStatus::Succeeded;
                        j.stage = "完成".into();
                    })
                    .await;
            }
        }
        Err(e) => {
            let cur = state
                .get_job(&job_id)
                .await
                .map(|j| j.status)
                .unwrap_or(JobStatus::Failed);
            if cur != JobStatus::Cancelled {
                state
                    .update_job(&job_id, |j| {
                        j.status = JobStatus::Failed;
                        j.error = Some(format!("{e:#}"));
                    })
                    .await;
            }
        }
    }
    // 持久化 status.json（便于重启后查看现场）
    if let Some(j) = state.get_job(&job_id).await {
        let p = store::status_json(&PathBuf::from(&j.workdir));
        let _ = std::fs::write(p, serde_json::to_vec_pretty(&j).unwrap_or_default());
    }
}

async fn run_inner(state: Arc<AppState>, id: &str) -> Result<()> {
    let job = state.get_job(id).await.context("任务不存在")?;
    let workdir = PathBuf::from(&job.workdir);
    let spec = job.spec.clone();
    let cann = state.cann.clone();

    if state.is_cancelled(id).await {
        return Ok(());
    }

    // 租约从选卡成功一直持有到 benchmark 和 msprof 全部结束。文件描述符关闭时
    // flock 由内核释放，因此正常返回、错误和进程崩溃都不会遗留死锁。
    let Some(lease) = acquire_device(&state, id, spec.device_id).await? else {
        return Ok(());
    };
    let dev = lease.device_id();

    let soc = spec
        .soc_version
        .clone()
        .or_else(|| state.cfg.soc_version.clone())
        .or_else(|| crate::cann::infer_soc(dev))
        .unwrap_or_else(|| "Ascend310P3".to_string());

    let onnx = store::onnx_path(&workdir);
    let om_base = store::om_base(&workdir);
    let om = store::om_path(&workdir);

    // ---------- 1. ATC 极限编译（带编译缓存）----------
    // 缓存键 = sha256( onnx_sha256 | soc | input_shape | extra_atc_flags )。
    // 缓存最终文件通过 rename 原子发布，允许不同设备上的任务并发编译。
    let onnx_sha = sha256_file(&onnx)?;
    let composite = format!(
        "{}\n{}\n{}\n{}",
        onnx_sha,
        soc,
        spec.input_shape.as_deref().unwrap_or(""),
        spec.extra_atc_flags.as_deref().unwrap_or(""),
    );
    let cache_key = hex_sha256(composite.as_bytes());
    let cache_om_path = store::cache_om(&state.storage, &cache_key);
    let cached = !spec.no_cache && cache_om_path.exists();

    let compile = if cached {
        state
            .update_job(id, |j| {
                j.status = JobStatus::Compiling;
                j.stage = "命中编译缓存，跳过 ATC".into();
            })
            .await;
        std::fs::copy(&cache_om_path, &om)
            .with_context(|| format!("从缓存复制 OM 失败: {}", cache_om_path.display()))?;
        CompileResult {
            duration_ms: 0,
            soc_version: soc.clone(),
            om_path: om.to_string_lossy().into_owned(),
            cached: true,
        }
    } else {
        state
            .update_job(id, |j| {
                j.status = JobStatus::Compiling;
                j.stage = "atc: 极限编译中".into();
            })
            .await;
        let atc_cmd = tools::build_atc_cmd(
            &onnx,
            &om_base,
            &soc,
            spec.input_shape.as_deref(),
            spec.extra_atc_flags.as_deref(),
        );
        let t0 = Instant::now();
        tools::run_logged(
            &cann.set_env,
            &atc_cmd,
            Some(&workdir),
            Some(&store::atc_log(&workdir)),
            "atc",
        )
        .await?;
        let cr = CompileResult {
            duration_ms: t0.elapsed().as_millis() as u64,
            soc_version: soc.clone(),
            om_path: om.to_string_lossy().into_owned(),
            cached: false,
        };
        // 先写任务专属临时文件再 rename，避免多卡并发任务观察到半写入的缓存。
        if !spec.no_cache {
            let _ = std::fs::create_dir_all(store::cache_dir(&state.storage, &cache_key));
            let cache_tmp =
                store::cache_dir(&state.storage, &cache_key).join(format!("model.{id}.tmp"));
            if std::fs::copy(&om, &cache_tmp).is_ok()
                && std::fs::rename(&cache_tmp, &cache_om_path).is_ok()
            {
                let manifest = json!({
                    "key": &cache_key,
                    "source_sha256": &onnx_sha,
                    "soc_version": &soc,
                    "input_shape": spec.input_shape,
                    "extra_atc_flags": spec.extra_atc_flags,
                    "om_size": std::fs::metadata(&om).map(|m| m.len()).unwrap_or(0),
                });
                let manifest_path = store::cache_manifest(&state.storage, &cache_key);
                let manifest_tmp = manifest_path.with_extension(format!("{id}.tmp"));
                if std::fs::write(
                    &manifest_tmp,
                    serde_json::to_vec_pretty(&manifest).unwrap_or_default(),
                )
                .is_ok()
                {
                    let _ = std::fs::rename(manifest_tmp, manifest_path);
                }
            } else {
                let _ = std::fs::remove_file(cache_tmp);
            }
        }
        cr
    };
    if state.is_cancelled(id).await {
        mark_cancelled(&state, id).await;
        return Ok(());
    }

    // ---------- 2. caliper-runner 基准 ----------
    state
        .update_job(id, |j| {
            j.status = JobStatus::Benchmarking;
            j.stage = "caliper-runner: 基准中".into();
        })
        .await;
    let runner_cmd = tools::build_runner_cmd(
        &state.runner,
        &om,
        dev,
        spec.iters,
        spec.warmup,
        &cann.libascendcl,
    );
    let stdout = tools::run_captured_stdout(
        &cann.set_env,
        &runner_cmd,
        Some(&workdir),
        Some(&store::runner_log(&workdir)),
        "caliper-runner",
    )
    .await?;
    let bench: BenchmarkResult =
        serde_json::from_str(stdout.trim()).context("解析 caliper-runner 输出失败")?;
    let _ = std::fs::write(
        store::bench_json(&workdir),
        serde_json::to_vec_pretty(&bench)?,
    );
    if state.is_cancelled(id).await {
        mark_cancelled(&state, id).await;
        return Ok(());
    }

    // ---------- 3. msprof 取证 ----------
    state
        .update_job(id, |j| {
            j.status = JobStatus::Profiling;
            j.stage = "msprof: 取证中".into();
        })
        .await;
    let mi = spec.msprof_iters.min(spec.iters.max(1));
    let msprof_out = store::msprof_dir(&workdir);
    let _ = std::fs::remove_dir_all(&msprof_out);
    let msprof_cmd =
        tools::build_msprof_cmd(&state.runner, &om, dev, mi, &cann.libascendcl, &msprof_out);
    let t1 = Instant::now();
    tools::run_logged(
        &cann.set_env,
        &msprof_cmd,
        Some(&workdir),
        Some(&store::msprof_log(&workdir)),
        "msprof",
    )
    .await?;
    let tgz = store::msprof_tgz(&workdir);
    tar_msprof(&workdir, &tgz).await?;
    let profile = ProfileResult {
        duration_ms: t1.elapsed().as_millis() as u64,
        msprof_dir: msprof_out.to_string_lossy().into_owned(),
        msprof_tar_gz: tgz.to_string_lossy().into_owned(),
    };

    // ---------- 4. 汇总 ----------
    let artifacts = store::list_artifacts(&workdir);
    let result = JobResult {
        compile,
        benchmark: Some(bench),
        profile,
        artifacts,
    };
    let _ = std::fs::write(
        store::result_json(&workdir),
        serde_json::to_vec_pretty(&result)?,
    );
    state.update_job(id, |j| j.result = Some(result)).await;
    drop(lease);
    Ok(())
}

async fn acquire_device(
    state: &Arc<AppState>,
    id: &str,
    requested: Option<i32>,
) -> Result<Option<DeviceLease>> {
    let candidates = state.devices.candidates(requested)?;
    loop {
        if state.is_cancelled(id).await {
            return Ok(None);
        }

        match state.devices.try_acquire(candidates.clone()).await? {
            TryAcquire::Acquired(lease) => {
                let device_id = lease.device_id();
                state
                    .update_job(id, |j| {
                        j.assigned_device_id = Some(device_id);
                        j.stage = format!("已独占 NPU {device_id}");
                    })
                    .await;
                return Ok(Some(lease));
            }
            TryAcquire::Unavailable(reason) => {
                state
                    .update_job(id, |j| {
                        j.status = JobStatus::Queued;
                        j.stage = if reason.is_empty() {
                            "等待空闲 NPU".into()
                        } else {
                            format!("等待空闲 NPU（{reason}）")
                        };
                    })
                    .await;
            }
        }
        tokio::time::sleep(state.devices.poll_interval()).await;
    }
}

async fn mark_cancelled(state: &Arc<AppState>, id: &str) {
    state
        .update_job(id, |j| {
            j.status = JobStatus::Cancelled;
            j.stage = "执行中取消".into();
        })
        .await;
}

async fn tar_msprof(workdir: &Path, tgz: &Path) -> Result<()> {
    let status = Command::new("tar")
        .arg("-czf")
        .arg(tgz)
        .arg("-C")
        .arg(workdir)
        .arg("msprof")
        .stderr(Stdio::piped())
        .status()
        .await
        .context("执行 tar 打包 msprof 失败")?;
    if !status.success() {
        anyhow::bail!("tar msprof 失败 exit={:?}", status.code());
    }
    Ok(())
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{:02x}", b)).collect()
}

fn sha256_file(path: &Path) -> Result<String> {
    let data = std::fs::read(path).with_context(|| format!("读取文件失败: {}", path.display()))?;
    Ok(hex_sha256(&data))
}
