//! libascendcl 的原始 FFI：用 libloading 动态加载，不在编译期链接 libascendcl，
//! 因此 crate 在任何机器上都能编译，只有运行时才要求 libascendcl.so 存在。
//!
//! 签名严格对照 CANN 8.5 头文件（acl_rt.h / acl_mdl.h / acl_base_rt.h）：
//!   - aclmdlIODims: { char name[128]; size_t dimCount; int64_t dims[128]; }  (ACL_MAX_DIM_CNT=128)
//!   - 添加 dataset 项的函数叫 aclmdlAddDatasetBuffer（非 AddDatasetItem）
//!   - aclrtMemMallocPolicy: ACL_MEM_MALLOC_HUGE_FIRST = 0

use anyhow::Context;
use libloading::Library;
use std::ffi::c_void;
use std::path::Path;

pub type Error = i32;
pub const OK: Error = 0;
/// aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST
pub const MEM_HUGE_FIRST: i32 = 0;

pub type Handle = *mut c_void;

/// 对应 C 的 aclmdlIODims。布局必须与头文件完全一致（共 1160 字节）。
#[repr(C)]
pub struct IODims {
    pub name: [u8; 128],
    pub dim_count: usize,
    pub dims: [i64; 128],
}

impl Default for IODims {
    fn default() -> Self {
        Self {
            name: [0u8; 128],
            dim_count: 0,
            dims: [0i64; 128],
        }
    }
}

impl IODims {
    pub fn shape(&self) -> Vec<u64> {
        let n = self.dim_count.min(128);
        self.dims[..n].iter().map(|&d| d.max(0) as u64).collect()
    }
}

// 函数指针类型别名，提升可读性。
pub type FnInit = unsafe extern "C" fn(*const u8) -> Error;
pub type FnFinalize = unsafe extern "C" fn() -> Error;
pub type FnSetDev = unsafe extern "C" fn(i32) -> Error;
pub type FnResetDev = unsafe extern "C" fn(i32) -> Error;
pub type FnLoad = unsafe extern "C" fn(*const u8, *mut u32) -> Error;
pub type FnUnload = unsafe extern "C" fn(u32) -> Error;
pub type FnCreateDesc = unsafe extern "C" fn() -> Handle;
pub type FnDestroyDesc = unsafe extern "C" fn(Handle) -> Error;
pub type FnGetDesc = unsafe extern "C" fn(Handle, u32) -> Error;
pub type FnGetNum = unsafe extern "C" fn(Handle) -> usize;
pub type FnGetSize = unsafe extern "C" fn(Handle, usize) -> usize;
pub type FnGetDims = unsafe extern "C" fn(Handle, usize, *mut IODims) -> Error;
pub type FnCreateDataset = unsafe extern "C" fn() -> Handle;
pub type FnDestroyDataset = unsafe extern "C" fn(Handle) -> Error;
pub type FnAddDatasetBuffer = unsafe extern "C" fn(Handle, Handle) -> Error;
pub type FnCreateDataBuffer = unsafe extern "C" fn(*mut c_void, usize) -> Handle;
pub type FnDestroyDataBuffer = unsafe extern "C" fn(Handle) -> Error;
pub type FnMalloc = unsafe extern "C" fn(*mut *mut c_void, usize, i32) -> Error;
pub type FnFree = unsafe extern "C" fn(*mut c_void) -> Error;
pub type FnMemset = unsafe extern "C" fn(*mut c_void, usize, i32, usize) -> Error;
pub type FnExecute = unsafe extern "C" fn(u32, Handle, Handle) -> Error;

/// 加载并持有所有用到的 ACL 符号。`_lib` 必须存活到所有符号使用结束。
pub struct Ffi {
    _lib: Library,
    pub init: FnInit,
    pub finalize: FnFinalize,
    pub rt_set_device: FnSetDev,
    pub rt_reset_device: FnResetDev,
    pub mdl_load_from_file: FnLoad,
    pub mdl_unload: FnUnload,
    pub mdl_create_desc: FnCreateDesc,
    pub mdl_destroy_desc: FnDestroyDesc,
    pub mdl_get_desc: FnGetDesc,
    pub mdl_get_num_inputs: FnGetNum,
    pub mdl_get_num_outputs: FnGetNum,
    pub mdl_get_input_size_by_index: FnGetSize,
    pub mdl_get_output_size_by_index: FnGetSize,
    pub mdl_get_input_dims: FnGetDims,
    pub mdl_get_output_dims: FnGetDims,
    pub mdl_create_dataset: FnCreateDataset,
    pub mdl_destroy_dataset: FnDestroyDataset,
    pub mdl_add_dataset_buffer: FnAddDatasetBuffer,
    pub create_data_buffer: FnCreateDataBuffer,
    pub destroy_data_buffer: FnDestroyDataBuffer,
    pub rt_malloc: FnMalloc,
    pub rt_free: FnFree,
    pub rt_memset: FnMemset,
    pub mdl_execute: FnExecute,
}

impl Ffi {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let lib = unsafe { Library::new(path) }
            .with_context(|| format!("加载 libascendcl 失败: {}", path.display()))?;
        // 取符号：*Symbol<T> 拷贝出 'static 的函数指针；lib 由 _lib 持有，进程退出前不卸载。
        let f = unsafe {
            Self {
                init: *lib.get::<FnInit>(b"aclInit\0")?,
                finalize: *lib.get::<FnFinalize>(b"aclFinalize\0")?,
                rt_set_device: *lib.get::<FnSetDev>(b"aclrtSetDevice\0")?,
                rt_reset_device: *lib.get::<FnResetDev>(b"aclrtResetDevice\0")?,
                mdl_load_from_file: *lib.get::<FnLoad>(b"aclmdlLoadFromFile\0")?,
                mdl_unload: *lib.get::<FnUnload>(b"aclmdlUnload\0")?,
                mdl_create_desc: *lib.get::<FnCreateDesc>(b"aclmdlCreateDesc\0")?,
                mdl_destroy_desc: *lib.get::<FnDestroyDesc>(b"aclmdlDestroyDesc\0")?,
                mdl_get_desc: *lib.get::<FnGetDesc>(b"aclmdlGetDesc\0")?,
                mdl_get_num_inputs: *lib.get::<FnGetNum>(b"aclmdlGetNumInputs\0")?,
                mdl_get_num_outputs: *lib.get::<FnGetNum>(b"aclmdlGetNumOutputs\0")?,
                mdl_get_input_size_by_index: *lib
                    .get::<FnGetSize>(b"aclmdlGetInputSizeByIndex\0")?,
                mdl_get_output_size_by_index: *lib
                    .get::<FnGetSize>(b"aclmdlGetOutputSizeByIndex\0")?,
                mdl_get_input_dims: *lib.get::<FnGetDims>(b"aclmdlGetInputDims\0")?,
                mdl_get_output_dims: *lib.get::<FnGetDims>(b"aclmdlGetOutputDims\0")?,
                mdl_create_dataset: *lib.get::<FnCreateDataset>(b"aclmdlCreateDataset\0")?,
                mdl_destroy_dataset: *lib.get::<FnDestroyDataset>(b"aclmdlDestroyDataset\0")?,
                mdl_add_dataset_buffer: *lib
                    .get::<FnAddDatasetBuffer>(b"aclmdlAddDatasetBuffer\0")?,
                create_data_buffer: *lib.get::<FnCreateDataBuffer>(b"aclCreateDataBuffer\0")?,
                destroy_data_buffer: *lib.get::<FnDestroyDataBuffer>(b"aclDestroyDataBuffer\0")?,
                rt_malloc: *lib.get::<FnMalloc>(b"aclrtMalloc\0")?,
                rt_free: *lib.get::<FnFree>(b"aclrtFree\0")?,
                rt_memset: *lib.get::<FnMemset>(b"aclrtMemset\0")?,
                mdl_execute: *lib.get::<FnExecute>(b"aclmdlExecute\0")?,
                _lib: lib,
            }
        };
        Ok(f)
    }
}
