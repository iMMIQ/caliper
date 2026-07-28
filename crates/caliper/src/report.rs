//! CLI 人类可读报告。保持 stdout 无颜色、无日志，便于重定向和终端查看。

use caliper_core::{ArtifactKind, IoDesc, Job, LatencyStats};
use std::fmt::Write;
use std::io::Write as IoWrite;

pub fn write_job(writer: &mut impl IoWrite, job: &Job, json: bool) -> anyhow::Result<()> {
    if json {
        serde_json::to_writer_pretty(&mut *writer, job)?;
        writer.write_all(b"\n")?;
    } else {
        writeln!(writer, "{}", format_job(job))?;
    }
    Ok(())
}

pub fn format_job(job: &Job) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Caliper Model Report");
    line(
        &mut out,
        "Model",
        job.onnx_name.as_deref().unwrap_or("<unknown>"),
    );
    line(&mut out, "Job ID", &job.id);
    line(&mut out, "Status", job.status.as_str());
    line(
        &mut out,
        "Device",
        &job.assigned_device_id
            .map(|device| device.to_string())
            .unwrap_or_else(|| "unassigned".into()),
    );
    line(&mut out, "Work directory", &job.workdir);

    if let Some(error) = &job.error {
        section(&mut out, "Error");
        let _ = writeln!(out, "  {error}");
    }

    let Some(result) = &job.result else {
        return out.trim_end().to_owned();
    };

    section(&mut out, "Compilation");
    line(&mut out, "SoC", &result.compile.soc_version);
    let compile_duration = if result.compile.cached {
        format!("{} ms (cache hit)", result.compile.duration_ms)
    } else {
        format!("{} ms", result.compile.duration_ms)
    };
    line(&mut out, "Duration", &compile_duration);
    line(&mut out, "OM", &result.compile.om_path);

    if let Some(benchmark) = &result.benchmark {
        section(&mut out, "Model Benchmark");
        line(
            &mut out,
            "Iterations",
            &format!("{} (warmup {})", benchmark.iterations, benchmark.warmup),
        );
        line(
            &mut out,
            "Inference latency",
            &format_latency(&benchmark.latency_us),
        );
        format_io(&mut out, "Inputs", &benchmark.inputs);
        format_io(&mut out, "Outputs", &benchmark.outputs);

        if let Some(transfer) = &benchmark.transfer {
            section(&mut out, "Transfer Benchmark");
            line(
                &mut out,
                "H2D",
                &format!(
                    "{}; total {}; effective bandwidth {:.6} GB/s",
                    format_latency(&transfer.h2d_latency_us),
                    format_bytes(transfer.input_bytes),
                    transfer.h2d_effective_bandwidth_gbps
                ),
            );
            line(
                &mut out,
                "D2H",
                &format!(
                    "{}; total {}; effective bandwidth {:.6} GB/s",
                    format_latency(&transfer.d2h_latency_us),
                    format_bytes(transfer.output_bytes),
                    transfer.d2h_effective_bandwidth_gbps
                ),
            );
        }
    }

    section(&mut out, "Profiling");
    line(
        &mut out,
        "Duration",
        &format!("{} ms", result.profile.duration_ms),
    );
    line(&mut out, "Archive", &result.profile.msprof_tar_gz);

    if !result.artifacts.is_empty() {
        section(&mut out, "Artifacts");
        for artifact in &result.artifacts {
            let _ = writeln!(
                out,
                "  {:<24} {:>14}  {}",
                artifact.name,
                format_bytes(artifact.size_bytes),
                artifact_kind(&artifact.kind)
            );
        }
    }

    out.trim_end().to_owned()
}

fn section(out: &mut String, title: &str) {
    let _ = writeln!(out, "\n{title}");
}

fn line(out: &mut String, label: &str, value: &str) {
    let _ = writeln!(out, "  {label}: {value}");
}

fn format_io(out: &mut String, label: &str, tensors: &[IoDesc]) {
    let _ = writeln!(out, "  {label}:");
    if tensors.is_empty() {
        let _ = writeln!(out, "    none");
        return;
    }
    for tensor in tensors {
        let shape = tensor
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join("x");
        let _ = writeln!(
            out,
            "    [{}] shape={}, size {}",
            tensor.index,
            if shape.is_empty() { "scalar" } else { &shape },
            format_bytes(tensor.size_bytes)
        );
    }
}

fn format_latency(stats: &LatencyStats) -> String {
    format!(
        "mean {:.3} us, p50 {:.3} us, p99 {:.3} us, min {:.3} us, max {:.3} us, stddev {:.3} us",
        stats.mean, stats.p50, stats.p99, stats.min, stats.max, stats.stddev
    )
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let value = bytes as f64;
    if value >= GIB {
        format!("{:.2} GiB ({} B)", value / GIB, bytes)
    } else if value >= MIB {
        format!("{:.2} MiB ({} B)", value / MIB, bytes)
    } else if value >= KIB {
        format!("{:.2} KiB ({} B)", value / KIB, bytes)
    } else {
        format!("{bytes} B")
    }
}

fn artifact_kind(kind: &ArtifactKind) -> &'static str {
    match kind {
        ArtifactKind::Onnx => "ONNX",
        ArtifactKind::Om => "OM",
        ArtifactKind::Log => "Log",
        ArtifactKind::Bench => "Benchmark",
        ArtifactKind::Atc => "ATC",
        ArtifactKind::Profile => "Profiling",
        ArtifactKind::Result => "Result",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliper_core::{
        BenchmarkResult, CompileResult, JobResult, JobSpec, JobStatus, ModelTransferResult,
        ProfileResult,
    };

    #[test]
    fn renders_a_readable_success_report() {
        let latency = LatencyStats {
            mean: 12.3456,
            p50: 12.0,
            p99: 15.0,
            min: 10.0,
            max: 16.0,
            stddev: 1.25,
        };
        let now = chrono::Utc::now();
        let job = Job {
            id: "job-1".into(),
            spec: JobSpec::default(),
            status: JobStatus::Succeeded,
            stage: "完成".into(),
            created_at: now,
            updated_at: now,
            error: None,
            result: Some(JobResult {
                compile: CompileResult {
                    duration_ms: 0,
                    soc_version: "Ascend310P3".into(),
                    om_path: "/tmp/model.om".into(),
                    cached: true,
                    ..Default::default()
                },
                benchmark: Some(BenchmarkResult {
                    iterations: 100,
                    warmup: 10,
                    device: 0,
                    latency_us: latency.clone(),
                    inputs: vec![IoDesc {
                        index: 0,
                        size_bytes: 602_112,
                        shape: vec![1, 3, 224, 224],
                    }],
                    outputs: vec![],
                    transfer: Some(ModelTransferResult {
                        iterations: 100,
                        warmup: 10,
                        input_bytes: 602_112,
                        output_bytes: 4_000,
                        h2d_latency_us: latency.clone(),
                        d2h_latency_us: latency,
                        h2d_effective_bandwidth_gbps: 9.69,
                        d2h_effective_bandwidth_gbps: 0.34,
                    }),
                }),
                profile: ProfileResult {
                    duration_ms: 123,
                    msprof_dir: "/tmp/msprof".into(),
                    msprof_tar_gz: "/tmp/msprof.tar.gz".into(),
                },
                artifacts: vec![],
            }),
            workdir: "/tmp/job-1".into(),
            onnx_name: Some("resnet.onnx".into()),
            assigned_device_id: Some(0),
        };

        let report = format_job(&job);
        assert!(report.contains("Model: resnet.onnx"));
        assert!(report.contains("Status: succeeded"));
        assert!(report.contains("Duration: 0 ms (cache hit)"));
        assert!(report.contains("shape=1x3x224x224"));
        assert!(report.contains("mean 12.346 us"));
        assert!(report.contains("effective bandwidth 9.690000 GB/s"));
        assert!(!report.starts_with('{'));

        let mut human = Vec::new();
        write_job(&mut human, &job, false).unwrap();
        assert!(String::from_utf8(human)
            .unwrap()
            .starts_with("Caliper Model Report\n"));

        let mut json = Vec::new();
        write_job(&mut json, &job, true).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(parsed["id"], "job-1");
        assert_eq!(parsed["status"], "succeeded");
    }
}
