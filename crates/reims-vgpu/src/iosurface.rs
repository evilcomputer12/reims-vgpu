//! [`IoSurfaceDevice`]: vmapple's second MMIO region, made from an [`IoSurfaceDeviceConfig`].
//!
//! The device registers itself as the process's latest IOSurface device, which a device made
//! with `use_iosurface_mapper` finds at its first use. Its interrupt runs on its own serial
//! worker; releasing the device from inside that callback ends the process, since its teardown
//! waits for that worker.

use std::sync::Arc;

use reims_vgpu_core::host::RaiseInterrupt;
use reims_vgpu_core::iosurface::HostDeviceHandle;
use reims_vgpu_core::iosurface::IoSurfaceHostDevice as CoreHostDevice;
use reims_vgpu_core::iosurface::descriptor::IoSurfaceHostDeviceDescriptor as CoreDescriptor;
use reims_vgpu_core::platform::Platform;

use crate::DefaultPlatform;
use crate::device::MemoryMap;
use crate::entry;
use crate::error::Error;
use crate::platform::ApiPlatform;
use crate::retained::Ours;

/// An IOSurface device's config.
#[derive(Clone, Default)]
pub struct IoSurfaceDeviceConfig {
    /// Required.
    pub memory_map: Option<MemoryMap>,
    /// `(vector)`, on the device's own worker. Without it, the device's first interrupt ends
    /// the process.
    pub raise_interrupt: Option<RaiseInterrupt>,
    /// The guest's page shift, as [`crate::DeviceConfig::guest_page_shift`]: 14 for an arm64
    /// guest's 16 KiB pages, 12 for an x86-64 guest's 4 KiB ones, 0 (the default) for the build's
    /// architecture's. The descriptor table, the descriptor pages, the surfaces' page tables and
    /// the ring count these pages. Any other value is refused ([`Error::InvalidArgument`]).
    pub guest_page_shift: u32,
}

impl std::fmt::Debug for IoSurfaceDeviceConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoSurfaceDeviceConfig")
            .field("memory_map", &self.memory_map)
            .field("raise_interrupt", &self.raise_interrupt.is_some())
            .field("guest_page_shift", &self.guest_page_shift)
            .finish()
    }
}

/// One reference to an IOSurface device: the last drop is its teardown, which waits for its
/// worker. Each call, and the last drop, runs as an entry ([`crate::entry`]).
pub struct IoSurfaceDevice<P: Platform = DefaultPlatform> {
    inner: Ours<HostDeviceHandle<P>>,
}

impl<P: Platform> std::fmt::Debug for IoSurfaceDevice<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("IoSurfaceDevice")
            .field(&std::ptr::from_ref::<CoreHostDevice<P>>(&self.inner))
            .finish()
    }
}

impl IoSurfaceDevice {
    /// A new IOSurface device on the process's platform ([`IoSurfaceDevice::with_platform`]).
    pub fn new(config: IoSurfaceDeviceConfig) -> Result<IoSurfaceDevice, Error> {
        IoSurfaceDevice::with_platform(DefaultPlatform::process(), config)
    }
}

impl<P: ApiPlatform> IoSurfaceDevice<P> {
    /// A new IOSurface device on `platform`, the process's latest. Refusals, in order:
    /// [`Error::InvalidArgument`] (a `guest_page_shift` other than 0, 12 and 14),
    /// [`Error::NoMemoryMap`], [`Error::MemoryMap`]; the closure is dropped before a refusal
    /// returns.
    pub fn with_platform(
        platform: Arc<P>,
        config: IoSurfaceDeviceConfig,
    ) -> Result<IoSurfaceDevice<P>, Error> {
        entry::run(|_| {
            let Some(guest_pages) = crate::device::guest_pages_of(config.guest_page_shift) else {
                return Err(Error::InvalidArgument);
            };
            let mut d = CoreDescriptor::new();
            d.guest_pages = guest_pages;
            d.memory_map_descriptor = config.memory_map.as_ref().map(MemoryMap::to_core);
            d.raise_interrupt = config.raise_interrupt;
            let made = CoreHostDevice::new(&platform, &d);
            drop(d);
            Ok(IoSurfaceDevice {
                inner: Ours::new(made?),
            })
        })
    }
}

impl<P: Platform> IoSurfaceDevice<P> {
    /// Another reference to this device.
    #[must_use]
    pub fn clone_ref(&self) -> IoSurfaceDevice<P> {
        IoSurfaceDevice {
            inner: Ours::new(self.inner.clone_ref()),
        }
    }

    /// Whether both are references to one device.
    pub fn ptr_eq(&self, other: &IoSurfaceDevice<P>) -> bool {
        self.inner.ptr_eq(&other.inner)
    }

    /// Core's device state, unretained: what a video device resolves the guest's mapping IDs
    /// through, so that its lookups never keep this device alive or run its teardown, as the
    /// graphics device's mapper service holds it.
    pub(crate) fn core_weak(
        &self,
    ) -> std::sync::Weak<reims_vgpu_core::iosurface::HostDeviceCore<P>> {
        self.inner.core_weak()
    }

    /// Core's device (tests).
    #[cfg(any(test, feature = "testkit"))]
    pub fn core(&self) -> &HostDeviceHandle<P> {
        &self.inner
    }

    /// A 64-bit read at `offset` in the device's registers (0x1000 to 0x1030); an offset that
    /// names no register reads all ones.
    pub fn mmio_read(&self, offset: u64) -> u64 {
        entry::run(|_| self.inner.mmio_read(offset))
    }

    /// A 64-bit write at `offset`; other offsets are ignored. May block on the device's worker.
    pub fn mmio_write(&self, offset: u64, value: u64) {
        entry::run(|_| {
            self.inner.mmio_write(offset, value);
        });
    }

    /// Re-create the IOSurfaces the descriptor table still describes, after a resume.
    pub fn rebuild_surfaces(&self) {
        entry::run(|_| {
            self.inner.rebuild_surfaces();
        });
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::testkit::FakePlatform;

    use super::*;

    /// No memory map refuses, and the closure goes with the refusal
    #[test]
    fn no_memory_map_refuses() {
        let p = Arc::new(FakePlatform::new());
        let raise: RaiseInterrupt = Arc::new(|_| {});
        let c = IoSurfaceDeviceConfig {
            memory_map: None,
            raise_interrupt: Some(raise.clone()),
            guest_page_shift: 0,
        };
        assert_eq!(
            IoSurfaceDevice::with_platform(p, c).err(),
            Some(Error::NoMemoryMap)
        );
        assert_eq!(Arc::strong_count(&raise), 1);
    }

    /// Made over one page of host memory, it registers as the process's latest and answers
    /// core's registers
    #[test]
    fn made_over_a_memory_map() {
        let p = Arc::new(FakePlatform::new());
        let vm = reims_vgpu_core::platform::Platform::vm(&*p);
        let page = vm.page_size();
        let host = vm.allocate_anywhere(page).unwrap();
        let mut m = MemoryMap::new();
        // SAFETY: the fake VM's region lives as long as the platform.
        unsafe { m.push(0, page, host.as_ptr()) };
        let c = IoSurfaceDeviceConfig {
            memory_map: Some(m),
            raise_interrupt: None,
            guest_page_shift: 0,
        };
        let d = IoSurfaceDevice::with_platform(p.clone(), c).unwrap();
        let found = CoreHostDevice::latest_handle(&*p).unwrap();
        assert!(found.ptr_eq(d.core()));
        for off in (0x1000..0x1038).step_by(8) {
            assert_eq!(d.mmio_read(off), d.core().mmio_read(off), "{off:#x}");
        }
        let d2 = d.clone_ref();
        assert!(d.ptr_eq(&d2));
    }

    /// The guest page shift: 0 is the build's pages, 12 and 14 an x86-64 and an arm64 guest's
    /// whatever the build; any other value is refused first, the closure going with it
    #[test]
    fn guest_page_shift() {
        let device = |shift: u32, raise: Option<RaiseInterrupt>| {
            let p = Arc::new(FakePlatform::new());
            let vm = reims_vgpu_core::platform::Platform::vm(&*p);
            let host = vm.allocate_anywhere(0x4000).unwrap();
            let mut m = MemoryMap::new();
            // SAFETY: the fake VM's region lives as long as the platform.
            unsafe { m.push(0, 0x4000, host.as_ptr()) };
            let c = IoSurfaceDeviceConfig {
                memory_map: Some(m),
                raise_interrupt: raise,
                guest_page_shift: shift,
            };
            IoSurfaceDevice::with_platform(p, c)
        };
        for shift in [0, 12, 14] {
            assert!(device(shift, None).is_ok(), "{shift}");
        }
        let raise: RaiseInterrupt = Arc::new(|_| {});
        for shift in [1, 13, 16] {
            assert_eq!(
                device(shift, Some(raise.clone())).err(),
                Some(Error::InvalidArgument),
                "{shift}"
            );
        }
        assert_eq!(Arc::strong_count(&raise), 1);
    }
}
