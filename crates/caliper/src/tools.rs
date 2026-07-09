//! 子进程工具：统一通过 `source set_env.sh; <cmd>` 在已配好 CANN 环境的 bash 中执行，
//! 捕获 stdout/stderr 并写入日志文件。atc / msprof / runner 的命令行构建也在此。

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;

/// shell 单引号转义（路径中假定不含单引号）。
pub fn shq(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// 构造 `bash -c "source <set_env>; <inner>"`。cwd 用于把子进程的工作目录钉到 job 目录，
/// 这样 atc/msprof 等往 CWD 丢的副产品（如 fusion_result.json）会落在 job 目录里、随 job 清理，
/// 而不是污染服务启动目录。
fn sourced(set_env: &Path, inner: &str, cwd: Option<&Path>) -> Command {
    let script = format!("source {} ; {}", shq(&set_env.to_string_lossy()), inner);
    let mut c = Command::new("bash");
    c.arg("-c").arg(script);
    if let Some(d) = cwd {
        c.current_dir(d);
    }
    c
}

struct Captured {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    code: Option<i32>,
}

async fn run_capture(set_env: &Path, inner: &str, cwd: Option<&Path>, stage: &str) -> Result<Captured> {
    let mut cmd = sourced(set_env, inner, cwd);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let out = cmd
        .output()
        .await
        .with_context(|| format!("spawn 失败 [{stage}]（检查 bash 是否可用）"))?;
    Ok(Captured {
        stdout: out.stdout,
        stderr: out.stderr,
        code: out.status.code(),
    })
}

fn write_log(log: Option<&Path>, inner: &str, cap: &Captured) {
    let Some(p) = log else { return };
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut s = String::new();
    s.push_str(&format!("$ {inner}\n"));
    if !cap.stdout.is_empty() {
        s.push_str("--- stdout ---\n");
        s.push_str(&String::from_utf8_lossy(&cap.stdout));
        s.push('\n');
    }
    if !cap.stderr.is_empty() {
        s.push_str("--- stderr ---\n");
        s.push_str(&String::from_utf8_lossy(&cap.stderr));
        s.push('\n');
    }
    s.push_str(&format!("exit={:?}\n", cap.code));
    let _ = std::fs::write(p, s);
}

fn tail(s: &[u8], n: usize) -> String {
    let s = String::from_utf8_lossy(s);
    if s.len() <= n {
        s.into_owned()
    } else {
        let start = s.len() - n;
        let mut out = String::from("…");
        out.push_str(&s[start..]);
        out
    }
}

/// 执行并写日志；失败时带 stderr 尾部 bail。
pub async fn run_logged(
    set_env: &Path,
    inner: &str,
    cwd: Option<&Path>,
    log: Option<&Path>,
    stage: &str,
) -> Result<()> {
    let cap = run_capture(set_env, inner, cwd, stage).await?;
    write_log(log, inner, &cap);
    if !matches!(cap.code, Some(0)) {
        bail!(
            "{stage} 失败 exit={:?}\n{}",
            cap.code,
            tail(&cap.stderr, 4000)
        );
    }
    Ok(())
}

/// 执行并返回 stdout 文本（runner 用：stdout 是 JSON）。stderr 落日志。
pub async fn run_captured_stdout(
    set_env: &Path,
    inner: &str,
    cwd: Option<&Path>,
    log: Option<&Path>,
    stage: &str,
) -> Result<String> {
    let cap = run_capture(set_env, inner, cwd, stage).await?;
    write_log(log, inner, &cap);
    if !matches!(cap.code, Some(0)) {
        bail!(
            "{stage} 失败 exit={:?}\n{}",
            cap.code,
            tail(&cap.stderr, 4000)
        );
    }
    Ok(String::from_utf8_lossy(&cap.stdout).into_owned())
}

/// 构建 ATC 命令（极限优化预设：O3 + l2_optimize + tiling + 编译缓存）。
pub fn build_atc_cmd(
    onnx: &Path,
    om_base: &Path,
    soc: &str,
    input_shape: Option<&str>,
    extra: Option<&str>,
) -> String {
    let mut parts: Vec<String> = vec![
        "atc".into(),
        // ATC 的 framework 取数值：5 = ONNX（字符串 "onnx" 在 CANN 8.5 不被接受）
        "--framework=5".into(),
        format!("--model={}", shq(&onnx.to_string_lossy())),
        format!("--output={}", shq(&om_base.to_string_lossy())),
        format!("--soc_version={}", shq(soc)),
        // 极限优化预设
        "--oo_level=O3".into(),
        "--buffer_optimize=l2_optimize".into(),
        "--tiling_schedule_optimize=1".into(),
        "--op_compiler_cache_mode=enable".into(),
        "--log=error".into(),
    ];
    if let Some(s) = input_shape.filter(|s| !s.trim().is_empty()) {
        parts.push(format!("--input_shape={}", shq(s)));
    }
    if let Some(s) = extra.filter(|s| !s.trim().is_empty()) {
        parts.push(s.trim().to_string());
    }
    parts.join(" ")
}

/// 构建 caliper-runner 基准命令。
pub fn build_runner_cmd(
    runner: &Path,
    om: &Path,
    device: i32,
    iters: u32,
    warmup: u32,
    lib: &Path,
) -> String {
    format!(
        "{} --om {} --device {} --iters {} --warmup {} --lib {}",
        shq(&runner.to_string_lossy()),
        shq(&om.to_string_lossy()),
        device,
        iters,
        warmup,
        shq(&lib.to_string_lossy()),
    )
}

/// 构建 msprof 命令：以 runner 作为 --application，取证少量迭代。
pub fn build_msprof_cmd(
    runner: &Path,
    om: &Path,
    device: i32,
    iters: u32,
    lib: &Path,
    out_dir: &Path,
) -> String {
    // 注意：application 值内的路径【不加】内层引号。msprof 按空格朴素切分程序名，
    // 若内层带引号，'.../caliper-runner' 整个会被当成程序名而找不到。
    // 因此要求 runner/om/lib 路径不含空格。整体再用 shq 包一层交给 bash。
    let app = format!(
        "{} --om {} --device {} --iters {} --warmup 0 --lib {}",
        runner.to_string_lossy(),
        om.to_string_lossy(),
        device,
        iters,
        lib.to_string_lossy(),
    );
    format!(
        "msprof --application={} --output={} --ascendcl=on --task-time=on --ai-core=on",
        shq(&app),
        shq(&out_dir.to_string_lossy()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atc_cmd_extreme_opts() {
        let cmd = build_atc_cmd(
            Path::new("/tmp/m.onnx"),
            Path::new("/tmp/m"),
            "Ascend310P3",
            Some("input:1,3,224,224"),
            None,
        );
        assert!(cmd.contains("--oo_level=O3"));
        assert!(cmd.contains("--buffer_optimize=l2_optimize"));
        assert!(cmd.contains("--tiling_schedule_optimize=1"));
        assert!(cmd.contains("--soc_version='Ascend310P3'"));
        assert!(cmd.contains("--input_shape='input:1,3,224,224'"));
    }

    #[test]
    fn shq_quotes_spaces() {
        assert_eq!(shq("a b"), "'a b'");
    }
}
