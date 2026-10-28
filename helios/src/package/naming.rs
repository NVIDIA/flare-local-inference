//! Readable cache-entry names: `<platform>-tp<N>-<UTC timestamp>`.
//!
//! Names are display metadata only. Payload and package identities remain
//! content hashes and signatures; renaming an entry never changes them.

use std::time::{SystemTime, UNIX_EPOCH};

/// Platforms a package may be named for. These are the NVIDIA device
/// identifiers Flare already uses for revisions, not SM architecture names.
pub const PLATFORMS: [&str; 3] = ["gb10", "gb202", "gb300"];

/// Platform for one NVIDIA PCI id in NVML's `device<<16 | vendor` layout.
///
/// This table is the single source consumed by Flare's device detection and by
/// package export naming; do not duplicate the identifiers elsewhere.
pub const GB202_DEVICES: [u32; 9] = [
    0x2b85, 0x2b87, 0x2b8c, 0x2bb1, 0x2bb2, 0x2bb3, 0x2bb4, 0x2bb5, 0x2bb9,
];

pub fn platform_from_pci(pci_id: u32) -> Option<&'static str> {
    if pci_id & 0xffff != 0x10de {
        return None;
    }
    let device = pci_id >> 16;
    if device == 0x2e12 {
        return Some("gb10");
    }
    if device == 0x31c3 {
        return Some("gb300");
    }
    GB202_DEVICES.contains(&device).then_some("gb202")
}

/// UTC `YYYYMMDDHHMM` for one instant, without external crates.
pub fn utc_timestamp(at: SystemTime) -> String {
    let seconds = at
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or(0);
    let days = (seconds / 86_400) as i64;
    let minutes = (seconds % 86_400) / 60;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}{month:02}{day:02}{:02}{:02}",
        minutes / 60,
        minutes % 60
    )
}

/// Base entry name for one exported deployment.
pub fn candidate(platform: &str, tensor_parallel: u32, at: SystemTime) -> Option<String> {
    if !PLATFORMS.contains(&platform) || tensor_parallel == 0 {
        return None;
    }
    Some(format!(
        "{platform}-tp{tensor_parallel}-{}",
        utc_timestamp(at)
    ))
}

/// Compatibility label for packages that lack an authenticated device identity.
///
/// This does not widen [`candidate`]'s supported-device mapping. It keeps older
/// packages with only SM or custom deployment identifiers loadable without
/// guessing a device from architecture or the local host.
pub fn unknown_device_candidate(sizes: &[u32], at: SystemTime) -> Option<String> {
    if sizes.is_empty() || sizes.iter().any(|size| *size == 0) {
        return None;
    }
    let label = sizes
        .iter()
        .map(|size| format!("tp{size}"))
        .collect::<Vec<_>>()
        .join("+");
    Some(format!("unknown-device-{label}-{}", utc_timestamp(at)))
}

/// Base entry name for one package's authenticated deployment sizes.
///
/// Sizes are sorted and deduplicated. An unsupported or absent device identity
/// uses the explicit `unknown-device` label; architecture and the local host
/// never supply one.
pub fn deployment_candidate(
    platform: Option<&str>,
    sizes: &[u32],
    at: SystemTime,
) -> Option<String> {
    let mut sizes = sizes.to_vec();
    sizes.sort_unstable();
    sizes.dedup();
    if sizes.is_empty() || sizes.iter().any(|size| *size == 0) {
        return None;
    }
    let Some(platform) = platform.filter(|platform| PLATFORMS.contains(platform)) else {
        return unknown_device_candidate(&sizes, at);
    };
    if let [size] = sizes.as_slice() {
        return candidate(platform, *size, at);
    }
    let label = sizes
        .iter()
        .map(|size| format!("tp{size}"))
        .collect::<Vec<_>>()
        .join("+");
    Some(format!("{platform}-{label}-{}", utc_timestamp(at)))
}

/// Same-minute collision label: the second attempt is `-2`, then `-3`, and so on.
pub fn collision(base: &str, ordinal: u32) -> Option<String> {
    (ordinal >= 2).then(|| format!("{base}-{ordinal}"))
}

/// Howard Hinnant's civil-from-days algorithm, for UTC fields only.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn at(seconds: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(seconds)
    }

    #[test]
    fn timestamps_are_utc_and_minute_resolved() {
        assert_eq!(utc_timestamp(UNIX_EPOCH), "197001010000");
        // 2026-10-17T09:23:00Z
        assert_eq!(utc_timestamp(at(1_792_228_980)), "202610170923");
        // Leap day.
        assert_eq!(utc_timestamp(at(1_709_164_800)), "202402290000");
        // One minute later, same day.
        assert_eq!(utc_timestamp(at(1_792_229_040)), "202610170924");
    }

    #[test]
    fn deployments_name_by_platform_and_tensor_parallel() {
        assert_eq!(
            candidate("gb202", 1, at(1_792_228_980)).as_deref(),
            Some("gb202-tp1-202610170923")
        );
        assert_eq!(
            candidate("gb10", 8, at(1_792_228_980)).as_deref(),
            Some("gb10-tp8-202610170923")
        );
        assert!(candidate("sm120", 1, at(0)).is_none());
        assert!(candidate("gb202", 0, at(0)).is_none());
    }

    #[test]
    fn pci_ids_select_the_platform_identifier() {
        assert_eq!(platform_from_pci(0x2e12_10de), Some("gb10"));
        assert_eq!(platform_from_pci(0x31c3_10de), Some("gb300"));
        for device in GB202_DEVICES {
            assert_eq!(
                platform_from_pci(device << 16 | 0x10de),
                Some("gb202"),
                "{device:#06x}"
            );
        }
        assert_eq!(platform_from_pci(0x2bbc_1af4), None);
        assert_eq!(platform_from_pci(0x31c3_1af4), None);
        assert_eq!(platform_from_pci(0x1e02_10de), None);
    }

    #[test]
    fn unknown_device_labels_are_explicit_and_never_use_sm_names() {
        let base = unknown_device_candidate(&[1], at(1_792_228_980)).unwrap();
        assert_eq!(base, "unknown-device-tp1-202610170923");
        assert!(!base.contains("sm120"));
        let union = unknown_device_candidate(&[1, 2], at(1_792_228_980)).unwrap();
        assert_eq!(union, "unknown-device-tp1+tp2-202610170923");
        assert!(unknown_device_candidate(&[], at(0)).is_none());
        assert!(unknown_device_candidate(&[0], at(0)).is_none());
    }

    #[test]
    fn deployments_name_by_signed_platform_and_sorted_union_sizes() {
        let at = at(1_792_228_980);
        assert_eq!(
            deployment_candidate(Some("gb202"), &[2, 1], at).as_deref(),
            Some("gb202-tp1+tp2-202610170923")
        );
        assert_eq!(
            deployment_candidate(Some("sm120"), &[1], at).as_deref(),
            Some("unknown-device-tp1-202610170923")
        );
        assert_eq!(
            deployment_candidate(None, &[2, 1], at).as_deref(),
            Some("unknown-device-tp1+tp2-202610170923")
        );
        assert!(deployment_candidate(None, &[], at).is_none());
        assert!(deployment_candidate(Some("gb202"), &[0], at).is_none());
    }

    #[test]
    fn same_minute_collisions_get_numeric_suffixes() {
        let base = candidate("gb202", 1, at(1_792_228_980)).unwrap();
        assert!(collision(&base, 1).is_none());
        assert_eq!(collision(&base, 2).unwrap(), format!("{base}-2"));
        assert_eq!(collision(&base, 3).unwrap(), format!("{base}-3"));
    }
}
