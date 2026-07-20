//! ACL 的安全封装：负责 init / set device / load OM / 构造输入输出 dataset /
//! 执行 / 卸载 / finalize 的完整生命周期。runner 单线程同步调用，不需要 Send。

pub mod ffi;

use anyhow::{bail, Result};
use std::ffi::CString;
use std::path::Path;
use std::time::Instant;

use caliper_core::IoDesc;
use ffi::{
    Error, Ffi, Handle, IODims, MEMCPY_DEVICE_TO_HOST, MEMCPY_HOST_TO_DEVICE, MEM_HUGE_FIRST, OK,
};

/// 已加载模型的资源句柄，用于卸载时按序释放。
struct ModelHandles {
    id: u32,
    desc: Handle,
    input_ds: Handle,
    output_ds: Handle,
    data_bufs: Vec<Handle>,
    input_buffers: Vec<MemoryBuffer>,
    output_buffers: Vec<MemoryBuffer>,
}

struct MemoryBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

pub struct Acl {
    f: Ffi,
    inited: bool,
    device: Option<i32>,
    model: Option<ModelHandles>,
}

fn check(stage: &str, code: Error) -> Result<()> {
    if code == OK {
        Ok(())
    } else {
        bail!("ACL 错误 @ {stage}: code={code} (0x{code:08X})");
    }
}

impl Acl {
    /// 动态加载 libascendcl.so。
    pub fn open(lib_path: &Path) -> Result<Self> {
        Ok(Self {
            f: Ffi::load(lib_path)?,
            inited: false,
            device: None,
            model: None,
        })
    }

    /// aclInit 全进程只能调一次。config 传 null。
    pub fn init(&mut self) -> Result<()> {
        if self.inited {
            return Ok(());
        }
        check("aclInit", unsafe { (self.f.init)(std::ptr::null()) })?;
        self.inited = true;
        Ok(())
    }

    pub fn set_device(&mut self, device: i32) -> Result<()> {
        check("aclrtSetDevice", unsafe { (self.f.rt_set_device)(device) })?;
        self.device = Some(device);
        Ok(())
    }

    /// 加载 OM，构造输入/输出 dataset（输入零填充），返回 IO 描述。
    pub fn load_model(&mut self, om: &Path) -> Result<(Vec<IoDesc>, Vec<IoDesc>)> {
        if self.model.is_some() {
            bail!("已有模型加载，请先 unload");
        }
        let path_c = CString::new(om.to_string_lossy().as_bytes())
            .map_err(|e| anyhow::anyhow!("OM 路径含内部 NUL: {e}"))?;

        let mut model_id: u32 = 0;
        check("aclmdlLoadFromFile", unsafe {
            (self.f.mdl_load_from_file)(path_c.as_ptr() as *const u8, &mut model_id)
        })?;

        let desc = unsafe { (self.f.mdl_create_desc)() };
        check("aclmdlGetDesc", unsafe {
            (self.f.mdl_get_desc)(desc, model_id)
        })?;

        let n_in = unsafe { (self.f.mdl_get_num_inputs)(desc) };
        let n_out = unsafe { (self.f.mdl_get_num_outputs)(desc) };

        let mut data_bufs = Vec::new();
        let mut input_buffers = Vec::new();
        let mut output_buffers = Vec::new();

        let input_ds = unsafe { (self.f.mdl_create_dataset)() };
        let inputs = self.build_side(
            desc,
            n_in,
            true,
            input_ds,
            &mut data_bufs,
            &mut input_buffers,
        )?;

        let output_ds = unsafe { (self.f.mdl_create_dataset)() };
        let outputs = self.build_side(
            desc,
            n_out,
            false,
            output_ds,
            &mut data_bufs,
            &mut output_buffers,
        )?;

        self.model = Some(ModelHandles {
            id: model_id,
            desc,
            input_ds,
            output_ds,
            data_bufs,
            input_buffers,
            output_buffers,
        });
        Ok((inputs, outputs))
    }

    /// 为一侧（输入/输出）分配每个 buffer，挂到 dataset。
    fn build_side(
        &self,
        desc: Handle,
        count: usize,
        is_input: bool,
        dataset: Handle,
        data_bufs: &mut Vec<Handle>,
        device_buffers: &mut Vec<MemoryBuffer>,
    ) -> Result<Vec<IoDesc>> {
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let size = if is_input {
                unsafe { (self.f.mdl_get_input_size_by_index)(desc, i) }
            } else {
                unsafe { (self.f.mdl_get_output_size_by_index)(desc, i) }
            };
            if size == 0 {
                bail!(
                    "模型 {side} buffer[{i}] size=0（可能为动态形状模型，暂不支持）",
                    side = if is_input { "input" } else { "output" }
                );
            }
            let mut dev: *mut std::ffi::c_void = std::ptr::null_mut();
            check("aclrtMalloc", unsafe {
                (self.f.rt_malloc)(&mut dev, size, MEM_HUGE_FIRST)
            })?;
            if is_input {
                // 输入零填充即可（量时延不关心正确性）
                check("aclrtMemset", unsafe {
                    (self.f.rt_memset)(dev, size, 0, size)
                })?;
            }
            let buf = unsafe { (self.f.create_data_buffer)(dev, size) };
            check("aclmdlAddDatasetBuffer", unsafe {
                (self.f.mdl_add_dataset_buffer)(dataset, buf)
            })?;
            data_bufs.push(buf);
            device_buffers.push(MemoryBuffer { ptr: dev, size });

            let shape = self.query_dims(desc, i, is_input).unwrap_or_default();
            out.push(IoDesc {
                index: i,
                size_bytes: size as u64,
                shape,
            });
        }
        Ok(out)
    }

    fn query_dims(&self, desc: Handle, i: usize, is_input: bool) -> Result<Vec<u64>> {
        let mut iod = IODims::default();
        let code = if is_input {
            unsafe { (self.f.mdl_get_input_dims)(desc, i, &mut iod) }
        } else {
            unsafe { (self.f.mdl_get_output_dims)(desc, i, &mut iod) }
        };
        check("aclmdlGetIODims", code)?;
        Ok(iod.shape())
    }

    /// 同步执行一次推理。调用方在外层计时。
    pub fn execute(&self) -> Result<()> {
        let m = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("execute 前未加载模型"))?;
        check("aclmdlExecute", unsafe {
            (self.f.mdl_execute)(m.id, m.input_ds, m.output_ds)
        })
    }

    /// 采集一次模型请求全部输入 tensor 的 H2D，以及全部输出 tensor 的 D2H 时延。
    /// 每个 tensor 使用独立页锁定 host buffer，分配和释放不计入样本。
    pub fn measure_model_transfer_ns(
        &self,
        iterations: u32,
        warmup: u32,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        if iterations == 0 {
            bail!("iterations 必须大于 0");
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("measure_model_transfer_ns 前未加载模型"))?;
        if model.input_buffers.is_empty() || model.output_buffers.is_empty() {
            bail!("模型必须至少包含一个输入和一个输出");
        }

        let host_inputs = self.allocate_host_buffers(&model.input_buffers)?;
        let host_outputs = match self.allocate_host_buffers(&model.output_buffers) {
            Ok(buffers) => buffers,
            Err(error) => {
                let _ = self.free_host_buffers(&host_inputs);
                return Err(error);
            }
        };

        let measured = (|| -> Result<(Vec<f64>, Vec<f64>)> {
            let h2d = || {
                for (host, device) in host_inputs.iter().zip(&model.input_buffers) {
                    check("aclrtMemcpy(model H2D)", unsafe {
                        (self.f.rt_memcpy)(
                            device.ptr,
                            device.size,
                            host.ptr,
                            host.size,
                            MEMCPY_HOST_TO_DEVICE,
                        )
                    })?;
                }
                Ok::<(), anyhow::Error>(())
            };
            let d2h = || {
                for (host, device) in host_outputs.iter().zip(&model.output_buffers) {
                    check("aclrtMemcpy(model D2H)", unsafe {
                        (self.f.rt_memcpy)(
                            host.ptr,
                            host.size,
                            device.ptr,
                            device.size,
                            MEMCPY_DEVICE_TO_HOST,
                        )
                    })?;
                }
                Ok::<(), anyhow::Error>(())
            };

            for _ in 0..warmup {
                h2d()?;
            }
            let mut h2d_ns = Vec::with_capacity(iterations as usize);
            for _ in 0..iterations {
                let t0 = Instant::now();
                h2d()?;
                h2d_ns.push(t0.elapsed().as_nanos() as f64);
            }

            for _ in 0..warmup {
                d2h()?;
            }
            let mut d2h_ns = Vec::with_capacity(iterations as usize);
            for _ in 0..iterations {
                let t0 = Instant::now();
                d2h()?;
                d2h_ns.push(t0.elapsed().as_nanos() as f64);
            }
            Ok((h2d_ns, d2h_ns))
        })();

        let free_inputs = self.free_host_buffers(&host_inputs);
        let free_outputs = self.free_host_buffers(&host_outputs);
        match measured {
            Err(error) => Err(error),
            Ok(samples) => {
                free_inputs?;
                free_outputs?;
                Ok(samples)
            }
        }
    }

    fn allocate_host_buffers(&self, device_buffers: &[MemoryBuffer]) -> Result<Vec<MemoryBuffer>> {
        let mut host_buffers = Vec::with_capacity(device_buffers.len());
        for device in device_buffers {
            let mut host = std::ptr::null_mut();
            if let Err(error) = check("aclrtMallocHost", unsafe {
                (self.f.rt_malloc_host)(&mut host, device.size)
            }) {
                let _ = self.free_host_buffers(&host_buffers);
                return Err(error);
            }
            unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0xA5, device.size) };
            host_buffers.push(MemoryBuffer {
                ptr: host,
                size: device.size,
            });
        }
        Ok(host_buffers)
    }

    fn free_host_buffers(&self, buffers: &[MemoryBuffer]) -> Result<()> {
        let mut result = Ok(());
        for buffer in buffers {
            if let Err(error) = check("aclrtFreeHost", unsafe {
                (self.f.rt_free_host)(buffer.ptr)
            }) {
                if result.is_ok() {
                    result = Err(error);
                }
            }
        }
        result
    }

    /// 使用预先分配的页锁定 host 内存和 device 内存，采集同步 H2D/D2H 拷贝时延。
    /// 分配、初始化和释放均不计入样本。
    pub fn measure_transfer_ns(
        &self,
        size: usize,
        iterations: u32,
        warmup: u32,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        if self.device.is_none() {
            bail!("measure_transfer_ns 前未 set device");
        }
        if size == 0 {
            bail!("传输大小必须大于 0");
        }
        if iterations == 0 {
            bail!("iterations 必须大于 0");
        }

        let mut host = std::ptr::null_mut();
        check("aclrtMallocHost", unsafe {
            (self.f.rt_malloc_host)(&mut host, size)
        })?;

        let mut device = std::ptr::null_mut();
        if let Err(error) = check("aclrtMalloc", unsafe {
            (self.f.rt_malloc)(&mut device, size, MEM_HUGE_FIRST)
        }) {
            let _ = check("aclrtFreeHost", unsafe { (self.f.rt_free_host)(host) });
            return Err(error);
        }

        // 提前触碰全部 host 页，避免缺页开销落入首次拷贝。
        unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0xA5, size) };

        let measured = (|| -> Result<(Vec<f64>, Vec<f64>)> {
            let h2d = || {
                check("aclrtMemcpy(H2D)", unsafe {
                    (self.f.rt_memcpy)(device, size, host, size, MEMCPY_HOST_TO_DEVICE)
                })
            };
            let d2h = || {
                check("aclrtMemcpy(D2H)", unsafe {
                    (self.f.rt_memcpy)(host, size, device, size, MEMCPY_DEVICE_TO_HOST)
                })
            };

            for _ in 0..warmup {
                h2d()?;
            }
            let mut h2d_ns = Vec::with_capacity(iterations as usize);
            for _ in 0..iterations {
                let t0 = Instant::now();
                h2d()?;
                h2d_ns.push(t0.elapsed().as_nanos() as f64);
            }

            for _ in 0..warmup {
                d2h()?;
            }
            let mut d2h_ns = Vec::with_capacity(iterations as usize);
            for _ in 0..iterations {
                let t0 = Instant::now();
                d2h()?;
                d2h_ns.push(t0.elapsed().as_nanos() as f64);
            }
            Ok((h2d_ns, d2h_ns))
        })();

        let free_device = check("aclrtFree", unsafe { (self.f.rt_free)(device) });
        let free_host = check("aclrtFreeHost", unsafe { (self.f.rt_free_host)(host) });
        match measured {
            Err(error) => Err(error),
            Ok(samples) => {
                free_device?;
                free_host?;
                Ok(samples)
            }
        }
    }

    /// 卸载模型并释放资源（best-effort，忽略个别错误以尽量清理）。
    pub fn unload_model(&mut self) {
        if let Some(m) = self.model.take() {
            for b in m.data_bufs {
                let _ = check("aclDestroyDataBuffer", unsafe {
                    (self.f.destroy_data_buffer)(b)
                });
            }
            for buffer in m.input_buffers.into_iter().chain(m.output_buffers) {
                if !buffer.ptr.is_null() {
                    let _ = check("aclrtFree", unsafe { (self.f.rt_free)(buffer.ptr) });
                }
            }
            let _ = check("aclmdlDestroyDataset", unsafe {
                (self.f.mdl_destroy_dataset)(m.input_ds)
            });
            let _ = check("aclmdlDestroyDataset", unsafe {
                (self.f.mdl_destroy_dataset)(m.output_ds)
            });
            let _ = check("aclmdlDestroyDesc", unsafe {
                (self.f.mdl_destroy_desc)(m.desc)
            });
            let _ = check("aclmdlUnload", unsafe { (self.f.mdl_unload)(m.id) });
        }
    }

    /// 收尾：reset device + finalize（best-effort）。
    pub fn shutdown(&mut self) {
        self.unload_model();
        if let Some(dev) = self.device.take() {
            let _ = check("aclrtResetDevice", unsafe { (self.f.rt_reset_device)(dev) });
        }
        if self.inited {
            let _ = check("aclFinalize", unsafe { (self.f.finalize)() });
            self.inited = false;
        }
    }
}

impl Drop for Acl {
    fn drop(&mut self) {
        self.shutdown();
    }
}
