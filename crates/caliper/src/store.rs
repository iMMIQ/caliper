//! 任务工作目录布局与产物枚举。

use caliper_core::{Artifact, ArtifactKind};
use std::path::{Path, PathBuf};

pub fn jobs_root(storage: &Path) -> PathBuf {
    storage.join("jobs")
}
pub fn job_dir(storage: &Path, id: &str) -> PathBuf {
    jobs_root(storage).join(id)
}

pub fn onnx_path(dir: &Path) -> PathBuf {
    dir.join("model.onnx")
}
pub fn om_path(dir: &Path) -> PathBuf {
    dir.join("model.om")
}
/// ATC 的 --output 基名（无扩展名），atc 会生成 model.om。
pub fn om_base(dir: &Path) -> PathBuf {
    dir.join("model")
}
pub fn atc_log(dir: &Path) -> PathBuf {
    dir.join("atc.log")
}
pub fn atc_pbtxt_dir(dir: &Path) -> PathBuf {
    dir.join("atc-pbtxt")
}
pub fn atc_pbtxt_tgz(dir: &Path) -> PathBuf {
    dir.join("atc-pbtxt.tar.gz")
}
pub fn bench_json(dir: &Path) -> PathBuf {
    dir.join("bench.json")
}
pub fn runner_log(dir: &Path) -> PathBuf {
    dir.join("runner.log")
}
pub fn result_json(dir: &Path) -> PathBuf {
    dir.join("result.json")
}
pub fn status_json(dir: &Path) -> PathBuf {
    dir.join("status.json")
}
pub fn meta_json(dir: &Path) -> PathBuf {
    dir.join("meta.json")
}
pub fn msprof_dir(dir: &Path) -> PathBuf {
    dir.join("msprof")
}
pub fn msprof_log(dir: &Path) -> PathBuf {
    dir.join("msprof.log")
}
pub fn msprof_tgz(dir: &Path) -> PathBuf {
    dir.join("msprof.tar.gz")
}

// ---- 编译缓存：<storage>/cache/<key>/{model.om, atc-pbtxt.tar.gz, manifest.json} ----
pub fn cache_root(storage: &Path) -> PathBuf {
    storage.join("cache")
}
pub fn cache_dir(storage: &Path, key: &str) -> PathBuf {
    cache_root(storage).join(key)
}
pub fn cache_om(storage: &Path, key: &str) -> PathBuf {
    cache_dir(storage, key).join("model.om")
}
pub fn cache_atc_pbtxt_tgz(storage: &Path, key: &str) -> PathBuf {
    cache_dir(storage, key).join("atc-pbtxt.tar.gz")
}
pub fn cache_manifest(storage: &Path, key: &str) -> PathBuf {
    cache_dir(storage, key).join("manifest.json")
}

/// 枚举工作目录下已知产物（仅存在的）。
pub fn list_artifacts(dir: &Path) -> Vec<Artifact> {
    let entries: [(PathBuf, &str, ArtifactKind); 8] = [
        (onnx_path(dir), "model.onnx", ArtifactKind::Onnx),
        (om_path(dir), "model.om", ArtifactKind::Om),
        (atc_log(dir), "atc.log", ArtifactKind::Log),
        (atc_pbtxt_tgz(dir), "atc-pbtxt.tar.gz", ArtifactKind::Atc),
        (bench_json(dir), "bench.json", ArtifactKind::Bench),
        (runner_log(dir), "runner.log", ArtifactKind::Log),
        (result_json(dir), "result.json", ArtifactKind::Result),
        (msprof_tgz(dir), "msprof.tar.gz", ArtifactKind::Profile),
    ];
    entries
        .into_iter()
        .filter_map(|(p, name, kind)| {
            std::fs::metadata(&p).ok().map(|m| Artifact {
                name: name.to_string(),
                size_bytes: m.len(),
                kind,
            })
        })
        .collect()
}

/// 产物白名单（防路径穿越）。命中返回其在工作目录下的路径。
pub fn artifact_path(dir: &Path, name: &str) -> Option<PathBuf> {
    const ALLOWED: &[&str] = &[
        "model.onnx",
        "model.om",
        "atc.log",
        "atc-pbtxt.tar.gz",
        "bench.json",
        "runner.log",
        "result.json",
        "status.json",
        "meta.json",
        "msprof.tar.gz",
        "msprof.log",
    ];
    if ALLOWED.contains(&name) {
        Some(dir.join(name))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_and_allows_atc_pbtxt_archive() {
        let dir = test_dir("atc-artifact");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(atc_pbtxt_tgz(&dir), b"archive").unwrap();

        let artifacts = list_artifacts(&dir);
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].name, "atc-pbtxt.tar.gz");
        assert_eq!(artifacts[0].kind, ArtifactKind::Atc);
        assert_eq!(
            artifact_path(&dir, "atc-pbtxt.tar.gz"),
            Some(atc_pbtxt_tgz(&dir))
        );
        assert!(artifact_path(&dir, "../atc-pbtxt.tar.gz").is_none());

        let _ = std::fs::remove_dir_all(dir);
    }

    fn test_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "caliper-store-test-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
