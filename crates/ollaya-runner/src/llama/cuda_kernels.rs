//! Whether llama.cpp's CUDA backend has kernels an NVIDIA GPU can run, checked before a model
//! loads (#42).
//!
//! The CUDA packs carry ggml-org's own `libggml-cuda` (b11146), which ships real GPU code (SASS)
//! for a few architectures and PTX for the rest. PTX is compiled by the driver when a kernel first
//! runs, and only a driver as new as the toolkit that wrote the PTX can do that. When neither
//! works, llama.cpp loads the model, then aborts the process at the first kernel launch, so the
//! runner looked healthy and died on the first question. Checking up front lets `auto` fall back
//! to the CPU with the reason, and an explicit device fail with it.
//!
//! The tables are `cuobjdump --list-elf` / `--list-ptx` of the pinned libraries (the Windows
//! build matches its Linux counterpart), and the toolkit version each library reports.

use std::ffi::c_int;
use std::path::Path;

/// One CUDA pack's `libggml-cuda`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kernels {
    /// The pack directory (`cuda_v13`).
    pub pack: &'static str,
    /// SASS: `(major, minor, arch_specific)`. Code for `x.y` runs on `x.z` with `z >= y`, except
    /// arch-specific code (`sm_120a`), which runs on exactly `x.y`.
    pub sass: &'static [(u32, u32, bool)],
    /// PTX: `(major, minor)`. Runs on any GPU of that compute capability or newer, JIT-compiled.
    pub ptx: &'static [(u32, u32)],
    /// The CUDA toolkit that wrote the PTX (`cuDriverGetVersion` form: 12080 = 12.8).
    pub ptx_toolkit: u32,
}

/// `lib/ollaya/cuda_v13`: CUDA 13.4 (ubuntu-cuda-13.4-x64, win-cuda-13.4-x64).
pub const CUDA13: Kernels = Kernels {
    pack: "cuda_v13",
    sass: &[(8, 6, false), (8, 9, false), (12, 0, true), (12, 1, true)],
    ptx: &[(7, 5), (8, 0), (9, 0)],
    ptx_toolkit: 13040,
};

/// `lib/ollaya/cuda_v12`: CUDA 12.8 (ubuntu-cuda-12.8-x64).
pub const CUDA12: Kernels = Kernels {
    pack: "cuda_v12",
    sass: &[(8, 6, false), (8, 9, false), (12, 0, true)],
    ptx: &[(5, 0), (6, 1), (7, 0), (7, 5), (8, 0), (9, 0)],
    ptx_toolkit: 12080,
};

/// The table for the CUDA backend at `path` (by its pack directory), if it is a known pack.
pub fn for_backend(path: &Path) -> Option<Kernels> {
    let pack = path.parent()?.file_name()?.to_str()?;
    [CUDA13, CUDA12].into_iter().find(|k| k.pack == pack)
}

fn version(v: u32) -> String {
    format!("{}.{}", v / 1000, (v % 1000) / 10)
}

impl Kernels {
    /// `Ok` when a GPU of compute capability `cc` can run this library's kernels on a driver of
    /// CUDA version `driver`; otherwise why not.
    pub fn check(&self, cc: (u32, u32), driver: u32) -> Result<(), String> {
        let (major, minor) = cc;
        let sass = self
            .sass
            .iter()
            .any(|&(ma, mi, a)| ma == major && if a { mi == minor } else { mi <= minor });
        if sass {
            return Ok(());
        }
        let Some(&(pma, pmi)) = self.ptx.iter().filter(|&&p| p <= cc).max() else {
            return Err(format!(
                "llama.cpp's CUDA kernels in {} do not support compute capability {major}.{minor}",
                self.pack
            ));
        };
        if driver >= self.ptx_toolkit {
            return Ok(());
        }
        Err(format!(
            "llama.cpp's CUDA kernels in {} cover compute capability {major}.{minor} only as \
             sm_{pma}{pmi} PTX, which the driver must compile, and that needs a driver for CUDA {} or \
             newer; this driver is for CUDA {}. Update the NVIDIA driver to use this GPU",
            self.pack,
            version(self.ptx_toolkit),
            version(driver)
        ))
    }
}

/// What the NVIDIA driver reports for one device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceInfo {
    pub compute_capability: (u32, u32),
    /// `cuDriverGetVersion`: 12030 for CUDA 12.3.
    pub driver: u32,
}

/// The compute capability of CUDA device `ordinal` and the driver's CUDA version, through the
/// driver API (`libcuda`, which every NVIDIA driver installs). `None` when that fails.
pub fn device_info(ordinal: i32) -> Option<DeviceInfo> {
    type Init = unsafe extern "C" fn(u32) -> c_int;
    type DriverVersion = unsafe extern "C" fn(*mut c_int) -> c_int;
    type DeviceGet = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
    type Attribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
    const CC_MAJOR: c_int = 75;
    const CC_MINOR: c_int = 76;
    let name = if cfg!(windows) {
        "nvcuda.dll"
    } else {
        "libcuda.so.1"
    };
    // SAFETY: libcuda's initializers have no side effects we depend on; every call below is the
    // documented driver API with valid out-pointers, and the library outlives the calls.
    unsafe {
        let lib = libloading::Library::new(name).ok()?;
        let init: libloading::Symbol<Init> = lib.get(b"cuInit\0").ok()?;
        let driver_version: libloading::Symbol<DriverVersion> =
            lib.get(b"cuDriverGetVersion\0").ok()?;
        let device_get: libloading::Symbol<DeviceGet> = lib.get(b"cuDeviceGet\0").ok()?;
        let attribute: libloading::Symbol<Attribute> = lib.get(b"cuDeviceGetAttribute\0").ok()?;
        if init(0) != 0 {
            return None;
        }
        let (mut driver, mut dev, mut major, mut minor) = (0, 0, 0, 0);
        if driver_version(&mut driver) != 0
            || device_get(&mut dev, ordinal) != 0
            || attribute(&mut major, CC_MAJOR, dev) != 0
            || attribute(&mut minor, CC_MINOR, dev) != 0
        {
            return None;
        }
        Some(DeviceInfo {
            compute_capability: (major as u32, minor as u32),
            driver: driver as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sass_runs_on_later_minors_except_arch_specific() {
        assert!(CUDA13.check((8, 6), 13000).is_ok());
        assert!(CUDA13.check((8, 9), 13000).is_ok());
        assert!(CUDA13.check((12, 0), 13000).is_ok());
        // sm_120a is exactly 12.0; a 12.1 GPU uses sm_121a.
        assert!(CUDA12.check((12, 1), 12030).is_err());
    }

    #[test]
    fn ptx_needs_a_driver_as_new_as_its_toolkit() {
        // #42: a GTX 1080 Ti (6.1) on driver 545 (CUDA 12.3) with the CUDA 12 pack.
        let e = CUDA12.check((6, 1), 12030).unwrap_err();
        assert!(
            e.contains("sm_61 PTX") && e.contains("CUDA 12.8") && e.contains("CUDA 12.3"),
            "{e}"
        );
        assert!(CUDA12.check((6, 1), 12080).is_ok());
        // A T4 (7.5) or an A100 (8.0) on R580 (13.0) with the CUDA 13 pack: PTX from 13.4.
        assert!(CUDA13.check((7, 5), 13000).is_err());
        assert!(CUDA13.check((8, 0), 13040).is_ok());
        // Newer than any SASS: the highest PTX at or below it (compute_90 for an sm_100).
        assert!(CUDA13.check((10, 0), 13040).is_ok());
    }

    #[test]
    fn older_than_every_kernel() {
        let e = CUDA13.check((6, 1), 13040).unwrap_err();
        assert!(e.contains("do not support compute capability 6.1"), "{e}");
    }

    #[test]
    fn pack_from_path() {
        let p = Path::new("/usr/lib/ollaya/cuda_v12/libggml-cuda.so");
        assert_eq!(for_backend(p), Some(CUDA12));
        assert_eq!(for_backend(Path::new("/x/y/libggml-cuda.so")), None);
    }
}
