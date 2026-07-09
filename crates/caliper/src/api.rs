//! HTTP API 路由与处理器。

use crate::state::AppState;
use crate::store;
use axum::{
    extract::{Multipart, Path, State},
    http::{header, StatusCode},
    response::{sse::Event, sse::KeepAlive, sse::Sse, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use caliper_core::{Artifact, Job, JobStatus};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/jobs", post(create_job).get(list_jobs))
        .route("/v1/jobs/:id", get(get_job).delete(cancel_job))
        .route("/v1/jobs/:id/events", get(job_events))
        .route("/v1/jobs/:id/artifacts", get(list_artifacts))
        .route("/v1/jobs/:id/artifacts/:name", get(get_artifact))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

type ApiError = (StatusCode, String);

async fn create_job(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, ApiError> {
    let mut spec: Option<caliper_core::JobSpec> = None;
    let mut onnx: Option<Vec<u8>> = None;
    let mut onnx_name: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("multipart: {e}")))?
    {
        match field.name().unwrap_or("") {
            "spec" => {
                let txt = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("read spec: {e}")))?;
                spec = Some(
                    serde_json::from_str(&txt)
                        .map_err(|e| (StatusCode::BAD_REQUEST, format!("parse spec: {e}")))?,
                );
            }
            "onnx" => {
                onnx_name = field.file_name().map(|s| s.to_string());
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("read onnx: {e}")))?;
                onnx = Some(bytes.to_vec());
            }
            _ => { /* ignore unknown */ }
        }
    }

    let onnx = onnx.ok_or((StatusCode::BAD_REQUEST, "缺少 onnx 字段".into()))?;
    let spec = spec.unwrap_or_default();

    let id = uuid::Uuid::new_v4().to_string();
    let workdir = store::job_dir(&state.storage, &id);
    std::fs::create_dir_all(&workdir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    std::fs::write(store::onnx_path(&workdir), &onnx)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut hasher = Sha256::new();
    hasher.update(&onnx);
    let hash: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();
    let meta = json!({"id": &id, "spec": &spec, "onnx_name": &onnx_name, "sha256": &hash});
    let _ = std::fs::write(
        store::meta_json(&workdir),
        serde_json::to_vec_pretty(&meta).unwrap_or_default(),
    );

    let now = chrono::Utc::now();
    let job = Job {
        id: id.clone(),
        spec: spec.clone(),
        status: JobStatus::Queued,
        stage: "排队中".into(),
        created_at: now,
        updated_at: now,
        error: None,
        result: None,
        workdir: workdir.to_string_lossy().into_owned(),
        onnx_name: onnx_name.clone(),
    };
    state.insert_job(job).await;
    state.register_cancel(&id).await;

    let st = state.clone();
    let id_run = id.clone();
    tokio::spawn(async move {
        crate::pipeline::run_pipeline(st, id_run).await;
    });

    let body = json!({
        "job_id": id,
        "status": "queued",
        "sha256": hash,
        "workdir": workdir.to_string_lossy(),
    });
    Ok((StatusCode::CREATED, Json(body)))
}

async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Job>, ApiError> {
    match state.get_job(&id).await {
        Some(j) => Ok(Json(j)),
        None => Err((StatusCode::NOT_FOUND, "job not found".into())),
    }
}

async fn list_jobs(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let jobs = state.list_jobs().await;
    let v: Vec<_> = jobs
        .into_iter()
        .map(|j| {
            json!({
                "id": j.id,
                "status": j.status.as_str(),
                "stage": j.stage,
                "created_at": j.created_at,
                "onnx_name": j.onnx_name,
            })
        })
        .collect();
    Json(v)
}

async fn cancel_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    if state.get_job(&id).await.is_none() {
        return Err((StatusCode::NOT_FOUND, "job not found".into()));
    }
    state.cancel(&id).await;
    Ok(Json(json!({"id": id, "cancelled": true})))
}

async fn list_artifacts(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Vec<Artifact>>, ApiError> {
    let j = state
        .get_job(&id)
        .await
        .ok_or((StatusCode::NOT_FOUND, "job not found".into()))?;
    let dir = PathBuf::from(&j.workdir);
    Ok(Json(store::list_artifacts(&dir)))
}

async fn get_artifact(
    State(state): State<Arc<AppState>>,
    Path((id, name)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let j = state
        .get_job(&id)
        .await
        .ok_or((StatusCode::NOT_FOUND, "job not found".into()))?;
    let dir = PathBuf::from(&j.workdir);
    let path = store::artifact_path(&dir, &name)
        .ok_or((StatusCode::NOT_FOUND, "unknown artifact".into()))?;
    let bytes = std::fs::read(&path).map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    let ct = content_type(&name);
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, ct.to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{}\"", name),
            ),
        ],
        bytes,
    )
        .into_response())
}

fn content_type(name: &str) -> &'static str {
    match name.rsplit('.').next() {
        Some("json") => "application/json",
        Some("log") => "text/plain; charset=utf-8",
        Some("om") | Some("onnx") => "application/octet-stream",
        Some("gz") => "application/gzip",
        _ => "application/octet-stream",
    }
}

/// SSE 进度流：GET /v1/jobs/:id/events
/// - `event: progress`：状态/阶段变化时推送快照 {status, stage, updated_at}
/// - `event: done`：进入终态时推送完整 Job
/// - `event: error`：任务不存在或超时
async fn job_events(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let s = async_stream::stream! {
        let mut last: Option<(String, String)> = None;
        // 上限约 1 小时（4500 × 800ms），防止僵尸流
        for _ in 0..4500 {
            match state.get_job(&id).await {
                None => {
                    yield Ok::<_, Infallible>(
                        Event::default().event("error").data("\"job not found\""),
                    );
                    return;
                }
                Some(j) => {
                    let sig = (j.status.as_str().to_string(), j.stage.clone());
                    if last.as_ref() != Some(&sig) {
                        let snap = serde_json::to_string(&json!({
                            "status": j.status.as_str(),
                            "stage": j.stage,
                            "updated_at": j.updated_at,
                        }))
                        .unwrap_or_else(|_| "{}".into());
                        yield Ok::<_, Infallible>(
                            Event::default().event("progress").data(snap),
                        );
                        last = Some(sig);
                    }
                    if j.status.is_terminal() {
                        let full = serde_json::to_string(&j).unwrap_or_else(|_| "{}".into());
                        yield Ok::<_, Infallible>(
                            Event::default().event("done").data(full),
                        );
                        return;
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(800)).await;
        }
        yield Ok::<_, Infallible>(Event::default().event("error").data("\"timeout\""));
    };
    Sse::new(s).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keepalive"),
    )
}
