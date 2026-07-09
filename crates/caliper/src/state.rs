//! 全局状态：任务表、按设备串行的锁、取消标志。

use crate::cann::Cann;
use crate::config::Config;
use caliper_core::{Job, JobId, JobStatus};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct AppState {
    pub cfg: Arc<Config>,
    pub cann: Arc<Cann>,
    pub runner: PathBuf,
    pub storage: PathBuf,
    pub jobs: Mutex<HashMap<JobId, Job>>,
    pub device_locks: Mutex<HashMap<i32, Arc<Mutex<()>>>>,
    pub cancel_flags: Mutex<HashMap<JobId, Arc<AtomicBool>>>,
}

impl AppState {
    pub fn new(cfg: Config, cann: Cann, runner: PathBuf, storage: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            cfg: Arc::new(cfg),
            cann: Arc::new(cann),
            runner,
            storage,
            jobs: Mutex::new(HashMap::new()),
            device_locks: Mutex::new(HashMap::new()),
            cancel_flags: Mutex::new(HashMap::new()),
        })
    }

    pub async fn insert_job(&self, job: Job) {
        self.jobs.lock().await.insert(job.id.clone(), job);
    }

    pub async fn get_job(&self, id: &str) -> Option<Job> {
        self.jobs.lock().await.get(id).cloned()
    }

    pub async fn list_jobs(&self) -> Vec<Job> {
        let g = self.jobs.lock().await;
        let mut v: Vec<Job> = g.values().cloned().collect();
        v.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        v
    }

    pub async fn update_job<F: FnOnce(&mut Job)>(&self, id: &str, f: F) {
        let mut g = self.jobs.lock().await;
        if let Some(j) = g.get_mut(id) {
            f(j);
            j.updated_at = chrono::Utc::now();
        }
    }

    pub async fn device_lock(&self, dev: i32) -> Arc<Mutex<()>> {
        let mut g = self.device_locks.lock().await;
        g.entry(dev)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    pub async fn register_cancel(&self, id: &str) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        self.cancel_flags
            .lock()
            .await
            .insert(id.to_string(), flag.clone());
        flag
    }

    pub async fn is_cancelled(&self, id: &str) -> bool {
        self.cancel_flags
            .lock()
            .await
            .get(id)
            .map(|f| f.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    pub async fn cancel(&self, id: &str) {
        if let Some(f) = self.cancel_flags.lock().await.get(id) {
            f.store(true, Ordering::SeqCst);
        }
        self.update_job(id, |j| {
            if !j.status.is_terminal() {
                j.status = JobStatus::Cancelled;
                j.stage = "用户取消".into();
            }
        })
        .await;
    }
}
