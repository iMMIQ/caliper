//! 多设备独占调度。
//!
//! `flock` 让同机的多个 Caliper 实例互斥；`npu-smi` 再检查没有遵守锁协议的
//! 外部任务。锁由文件描述符持有，进程崩溃或任务 unwind 后内核会自动释放。

use anyhow::{bail, Context, Result};
use fs2::FileExt;
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

pub struct DeviceManager {
    device_ids: Vec<i32>,
    lock_dir: PathBuf,
    poll_interval: Duration,
    require_idle: bool,
    cursor: AtomicUsize,
}

pub struct DeviceLease {
    device_id: i32,
    lock_file: File,
}

impl DeviceLease {
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl Drop for DeviceLease {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.lock_file);
    }
}

pub enum TryAcquire {
    Acquired(DeviceLease),
    Unavailable(String),
}

impl DeviceManager {
    pub fn new(
        configured_ids: Vec<i32>,
        lock_dir: PathBuf,
        poll_interval_ms: u64,
        require_idle: bool,
    ) -> Result<Self> {
        let mut ids = if configured_ids.is_empty() {
            discover_device_ids()
        } else {
            configured_ids
        };
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            bail!("未发现 NPU 设备；请配置 [devices].ids 或 --devices");
        }
        if ids.iter().any(|id| *id < 0) {
            bail!("设备 ID 不能为负数: {ids:?}");
        }
        ensure_shared_lock_dir(&lock_dir)?;
        // 服务启动时一次性创建并验证所有锁文件，避免任务在排队后才因权限错误失败。
        for device_id in &ids {
            open_lock_file(&lock_dir, *device_id)?;
        }
        Ok(Self {
            device_ids: ids,
            lock_dir,
            poll_interval: Duration::from_millis(poll_interval_ms.max(100)),
            require_idle,
            cursor: AtomicUsize::new(0),
        })
    }

    pub fn device_ids(&self) -> &[i32] {
        &self.device_ids
    }

    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    pub fn candidates(&self, requested: Option<i32>) -> Result<Vec<i32>> {
        if let Some(id) = requested {
            if !self.device_ids.contains(&id) {
                bail!("请求的设备 {id} 不在允许设备池 {:?} 中", self.device_ids);
            }
            return Ok(vec![id]);
        }
        let len = self.device_ids.len();
        let start = self.cursor.fetch_add(1, Ordering::Relaxed) % len;
        Ok((0..len)
            .map(|offset| self.device_ids[(start + offset) % len])
            .collect())
    }

    pub async fn try_acquire(&self, candidates: Vec<i32>) -> Result<TryAcquire> {
        let lock_dir = self.lock_dir.clone();
        let require_idle = self.require_idle;
        tokio::task::spawn_blocking(move || {
            let mut reasons = Vec::new();
            for device_id in candidates {
                let file = open_lock_file(&lock_dir, device_id)?;
                match file.try_lock_exclusive() {
                    Ok(()) => {}
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {
                        reasons.push(format!("{device_id}: 已被 Caliper 租用"));
                        continue;
                    }
                    Err(e) => return Err(e).context("获取设备租约锁失败"),
                }

                if require_idle {
                    match busy_devices() {
                        Ok(busy) if busy.contains(&device_id) => {
                            reasons.push(format!("{device_id}: npu-smi 检测到外部进程"));
                            let _ = FileExt::unlock(&file);
                            continue;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            reasons.push(format!("{device_id}: 无法确认空闲（{e:#}）"));
                            let _ = FileExt::unlock(&file);
                            continue;
                        }
                    }
                }
                return Ok(TryAcquire::Acquired(DeviceLease {
                    device_id,
                    lock_file: file,
                }));
            }
            Ok(TryAcquire::Unavailable(reasons.join("；")))
        })
        .await
        .context("设备调度线程异常")?
    }
}

fn ensure_shared_lock_dir(path: &Path) -> Result<()> {
    let existed = path.exists();
    match std::fs::create_dir_all(path) {
        Ok(()) => {
            if !existed {
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o1777))
                    .with_context(|| {
                        format!(
                            "设备锁目录 {} 必须可供所有目标用户访问；请由管理员执行 chmod 1777 {}",
                            path.display(),
                            path.display()
                        )
                    })?;
            }
        }
        Err(e) => return Err(e).with_context(|| format!("创建锁目录失败: {}", path.display())),
    }
    if !path.is_dir() {
        bail!("设备锁路径不是目录: {}", path.display());
    }
    Ok(())
}

fn open_lock_file(lock_dir: &Path, device_id: i32) -> Result<File> {
    let path = lock_dir.join(format!("device-{device_id}.lock"));
    match OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o666)
        .open(&path)
    {
        Ok(file) => {
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666))?;
            Ok(file)
        }
        Err(e) if e.kind() == ErrorKind::AlreadyExists => OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| {
                format!(
                    "打开设备锁失败: {}；请确认该文件权限为 0666",
                    path.display()
                )
            }),
        Err(e) => Err(e).with_context(|| format!("创建设备锁失败: {}", path.display())),
    }
}

fn discover_device_ids() -> Vec<i32> {
    if let Ok(out) = Command::new("npu-smi").args(["info", "-l"]).output() {
        if out.status.success() {
            let ids = parse_device_list(&String::from_utf8_lossy(&out.stdout));
            if !ids.is_empty() {
                return ids;
            }
        }
    }
    let mut ids = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/dev") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(suffix) = name.to_str().and_then(|s| s.strip_prefix("davinci")) else {
                continue;
            };
            if let Ok(id) = suffix.parse::<i32>() {
                ids.push(id);
            }
        }
    }
    ids
}

fn parse_device_list(text: &str) -> Vec<i32> {
    let mut ids = BTreeSet::new();
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        if !(lower.contains("npu id") || lower.contains("device id")) {
            continue;
        }
        if let Some(value) = line.split(':').nth(1) {
            if let Some(id) = value
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<i32>().ok())
            {
                ids.insert(id);
            }
        }
    }
    ids.into_iter().collect()
}

fn busy_devices() -> Result<BTreeSet<i32>> {
    let out = Command::new("npu-smi")
        .arg("info")
        .output()
        .context("执行 npu-smi info 失败")?;
    if !out.status.success() {
        bail!("npu-smi info exit={:?}", out.status.code());
    }
    parse_busy_devices(&String::from_utf8_lossy(&out.stdout))
}

fn parse_busy_devices(text: &str) -> Result<BTreeSet<i32>> {
    let mut busy = BTreeSet::new();
    let mut in_process_table = false;
    let mut recognized = false;
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("process id") && lower.contains("process name") {
            in_process_table = true;
            recognized = true;
            continue;
        }
        if lower.contains("no running processes found") {
            recognized = true;
            continue;
        }
        if !in_process_table || !line.contains('|') {
            continue;
        }
        let cols: Vec<&str> = line
            .split('|')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if cols.len() < 3 {
            continue;
        }
        let device = cols[0].parse::<i32>();
        let pid = cols[2].parse::<u32>();
        if let (Ok(device), Ok(pid)) = (device, pid) {
            if pid > 0 {
                busy.insert(device);
            }
        }
    }
    if !recognized {
        bail!("无法识别 npu-smi 进程表格式");
    }
    Ok(busy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_list() {
        let text = "Total Count : 2\nNPU ID : 0\nNPU ID : 3\n";
        assert_eq!(parse_device_list(text), vec![0, 3]);
    }

    #[test]
    fn candidate_selection_rotates_and_honors_request() {
        let dir = test_lock_dir("candidates");
        let manager = DeviceManager::new(vec![2, 0, 1, 1], dir.clone(), 100, false).unwrap();
        assert_eq!(manager.candidates(None).unwrap(), vec![0, 1, 2]);
        assert_eq!(manager.candidates(None).unwrap(), vec![1, 2, 0]);
        assert_eq!(manager.candidates(Some(2)).unwrap(), vec![2]);
        assert!(manager.candidates(Some(7)).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn parses_busy_process_table() {
        let text = r#"
| NPU | Chip | Process id | Process name | Process memory(MB) |
| 0   | 0    | 1234       | python3      | 100                |
| 2   | 0    | 5678       | app          | 200                |
"#;
        assert_eq!(parse_busy_devices(text).unwrap(), BTreeSet::from([0, 2]));
    }

    #[test]
    fn accepts_explicit_no_process_message() {
        let text = "No running processes found in NPU 0";
        assert!(parse_busy_devices(text).unwrap().is_empty());
    }

    #[test]
    fn unknown_process_format_fails_closed() {
        assert!(parse_busy_devices("some unexpected output").is_err());
    }

    #[tokio::test]
    async fn lease_is_exclusive_and_released_on_drop() {
        let dir = test_lock_dir("lease");
        let manager = DeviceManager::new(vec![7], dir.clone(), 100, false).unwrap();

        let first = match manager.try_acquire(vec![7]).await.unwrap() {
            TryAcquire::Acquired(lease) => lease,
            TryAcquire::Unavailable(reason) => panic!("first lease unavailable: {reason}"),
        };
        assert!(matches!(
            manager.try_acquire(vec![7]).await.unwrap(),
            TryAcquire::Unavailable(_)
        ));

        drop(first);
        assert!(matches!(
            manager.try_acquire(vec![7]).await.unwrap(),
            TryAcquire::Acquired(_)
        ));
        let _ = std::fs::remove_dir_all(dir);
    }

    fn test_lock_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "caliper-device-test-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
