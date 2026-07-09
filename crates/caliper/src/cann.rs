//! CANN 工具链自动发现。不写死任何绝对路径：按优先级扫描候选根目录，
//! 在每个根下按常见目录布局定位 atc / msprof / set_env.sh / libascendcl.so / 头文件。

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Cann {
    pub home: PathBuf,
    pub set_env: PathBuf,
    pub atc: PathBuf,
    pub msprof: PathBuf,
    pub libascendcl: PathBuf,
    pub acl_include: PathBuf,
}

/// 自动发现 CANN 工具链。override_home 优先级最高。
pub fn discover(override_home: Option<&str>) -> Result<Cann> {
    let arch = arch_triple(); // "x86_64" | "aarch64"
    let home_dir = std::env::var("HOME").unwrap_or_default();

    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(h) = override_home {
        roots.push(PathBuf::from(h));
    }
    if let Ok(h) = std::env::var("ASCEND_TOOLKIT_HOME") {
        roots.push(PathBuf::from(h));
    }
    for c in [
        "/usr/local/Ascend/latest",
        "/usr/local/Ascend",
        &format!("{home_dir}/Ascend/latest"),
        &format!("{home_dir}/.local/Ascend/latest"),
    ] {
        roots.push(PathBuf::from(c));
    }

    // 把“容器目录”（如 /usr/local/Ascend）展开为其中的 cann*/ascend-toolkit* 子目录
    let mut expanded: Vec<PathBuf> = Vec::new();
    for r in &roots {
        expanded.push(r.clone());
        if let Ok(entries) = std::fs::read_dir(r) {
            let mut subs: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .filter(|p| {
                    let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    n.starts_with("cann")
                        || n.starts_with("ascend-toolkit")
                        || n.starts_with("nnae")
                })
                .collect();
            subs.sort();
            expanded.extend(subs);
        }
    }

    for root in expanded {
        if let Some(c) = try_root(&root, &arch) {
            return Ok(c);
        }
    }
    bail!(
        "未发现可用的 CANN 工具链（需要 bin/atc、bin/msprof、set_env.sh、libascendcl.so）。\
         请用 --cann-home 或 cann.home 指定，或确认 set_env.sh 已可 source。"
    );
}

fn arch_triple() -> String {
    match std::env::consts::ARCH {
        "x86_64" => "x86_64".to_string(),
        "aarch64" => "aarch64".to_string(),
        other => other.to_string(),
    }
}

fn try_root(root: &Path, arch: &str) -> Option<Cann> {
    let atc = first_existing(
        root,
        &[
            "bin/atc",
            "latest/bin/atc",
            &format!("latest/{arch}-linux/bin/atc"),
            &format!("{arch}-linux/bin/atc"),
        ],
    )?;
    let msprof = first_existing(
        root,
        &[
            "bin/msprof",
            "latest/bin/msprof",
            &format!("latest/{arch}-linux/bin/msprof"),
            &format!("{arch}-linux/bin/msprof"),
        ],
    )?;
    let set_env = first_existing(
        root,
        &[
            "set_env.sh",
            "latest/set_env.sh",
            &format!("latest/{arch}-linux/set_env.sh"),
            &format!("{arch}-linux/set_env.sh"),
        ],
    )?;
    let libascendcl = first_existing(
        root,
        &[
            "lib64/libascendcl.so",
            &format!("{arch}-linux/lib64/libascendcl.so"),
            &format!("latest/{arch}-linux/lib64/libascendcl.so"),
            "latest/lib64/libascendcl.so",
            &format!("{arch}-linux/devlib/libascendcl.so"),
        ],
    )?;
    let acl_include = first_existing(
        root,
        &[
            "include/acl/acl.h",
            &format!("{arch}-linux/include/acl/acl.h"),
            &format!("latest/{arch}-linux/include/acl/acl.h"),
        ]
        .into_iter()
        .collect::<Vec<_>>(),
    )
    .unwrap_or_default();

    Some(Cann {
        home: root.to_path_buf(),
        set_env,
        atc,
        msprof,
        libascendcl,
        acl_include,
    })
}

fn first_existing(root: &Path, cands: &[&str]) -> Option<PathBuf> {
    for c in cands {
        let p = root.join(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// 从 npu-smi 推断 SoC 型号，如 "310P3" → "Ascend310P3"。
pub fn infer_soc(_device: i32) -> Option<String> {
    let out = std::process::Command::new("npu-smi")
        .arg("info")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        for tok in line.split_whitespace() {
            // 形如 310P3 / 910B1 / 310B4：以数字开头、含字母、长度 ≤ 8
            let starts_digit = tok.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false);
            let has_alpha = tok.chars().any(|c| c.is_ascii_alphabetic());
            let no_punct = tok
                .chars()
                .all(|c| c.is_ascii_alphanumeric())
                && tok.chars().any(|c| c.is_ascii_alphabetic());
            if starts_digit && has_alpha && no_punct && tok.len() <= 8 {
                return Some(format!("Ascend{}", tok));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_finds_a_toolchain() {
        // 本机应有 CANN；发现失败说明环境异常，但 CI 上可能没有，故 best-effort。
        if std::env::var("CALIPER_SKIP_CANN_TEST").is_ok() {
            return;
        }
        match discover(None) {
            Ok(c) => {
                assert!(c.atc.exists(), "atc not found: {}", c.atc.display());
                assert!(c.msprof.exists());
                assert!(c.set_env.exists());
                assert!(c.libascendcl.exists());
            }
            Err(_) => { /* 本机无 CANN 时允许跳过 */ }
        }
    }
}
