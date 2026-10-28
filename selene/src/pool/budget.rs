use super::PlannedRegion;

/// Remaining startup supply, after local tiers, runtime headroom and minimum caches.
pub struct Budget {
    pub host: std::result::Result<u64, String>,
    pub devices: Vec<(u64, bool)>,
}

impl Budget {
    pub(super) fn costs(
        mut bytes: impl Iterator<Item = u64>,
    ) -> std::result::Result<(u64, u64), String> {
        bytes.try_fold((0u64, 0u64), |(host, device), bytes| {
            Ok((
                host.checked_add(bytes).ok_or("pool host bytes overflow")?,
                device
                    .checked_add(bytes.div_ceil(512))
                    .ok_or("pool device bytes overflow")?,
            ))
        })
    }

    pub fn charges(&self, host: u64, device: u64) -> Vec<u64> {
        let integrated = self.devices.iter().filter(|(_, shared)| *shared).count() as u64;
        self.devices
            .iter()
            .map(|(_, shared)| {
                device.saturating_add(if *shared {
                    host.div_ceil(integrated)
                } else {
                    0
                })
            })
            .collect()
    }

    pub(super) fn approve(&self, plan: &[PlannedRegion]) -> std::result::Result<(), String> {
        let (host, device) = Self::costs(plan.iter().map(|region| region.bytes))?;
        let available_host = self
            .host
            .as_ref()
            .map_err(|error| format!("cannot budget pool host memory: {error}"))?;
        if host > *available_host {
            return Err(format!(
                "needs {host} host bytes and {device} device bytes per GPU; host budget has {} bytes left",
                available_host
            ));
        }
        for (rank, ((available, _), required)) in self
            .devices
            .iter()
            .zip(self.charges(host, device))
            .enumerate()
        {
            if required > *available {
                return Err(format!(
                    "needs {host} host bytes and {device} device bytes per GPU; GPU {rank} needs {required} bytes including shared host memory, but its budget has {available} bytes left"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
