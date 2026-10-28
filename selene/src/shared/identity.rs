//! Identities both ends of the pool protocol derive and compare: the scope
//! id, the instance id and the process nonce.
//!
//! The byte layouts below are the contract. They are written out here, not
//! derived from a message encoder, so the manager and every instance hash the
//! same bytes. All integers are little-endian.

use blake2::{
    Blake2b, Digest,
    digest::consts::{U16, U32},
};

use crate::{CacheGeometry, Error, Layout, Plane, Result};

/// The plane and row sizes of one rank, as the GPU geometry declares them:
/// unpadded unit sizes. A record pads each to a 4 KiB block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankPlanes {
    pub plane_bytes: Vec<u64>,
    pub row_bytes: u64,
}

/// The record geometry of one model package on one rank layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub page_tokens: u64,
    pub ranks: Vec<RankPlanes>,
    pub page_record_bytes: u64,
    pub row_record_bytes: u64,
}

/// What a session may share. Records are shared only between identical scopes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    pub model_identity: [u8; 32],
    pub layout_fingerprint: [u8; 16],
    pub geometry: Geometry,
}

impl Scope {
    /// The scope of the records a model package writes on `geometry`, one
    /// element per rank, under `layout_fingerprint` (see
    /// [`crate::geometry::fingerprint`] and `shard_fingerprint`).
    pub fn new(
        model_identity: [u8; 32],
        layout_fingerprint: [u8; 16],
        geometry: &[CacheGeometry],
    ) -> Result<Self> {
        let layout = Layout::new(geometry)?;
        Ok(Self {
            model_identity,
            layout_fingerprint,
            geometry: Geometry {
                page_tokens: layout.page_tokens,
                ranks: geometry
                    .iter()
                    .map(|rank| RankPlanes {
                        plane_bytes: rank.planes.iter().map(|plane| plane.unit_bytes).collect(),
                        row_bytes: rank.row.unit_bytes,
                    })
                    .collect(),
                page_record_bytes: layout.page_record_bytes,
                row_record_bytes: layout.row_record_bytes,
            },
        })
    }

    /// A cache geometry with this scope's sizes (names and addresses are
    /// placeholders), for building record headers.
    pub(crate) fn record_geometry(&self) -> Vec<CacheGeometry> {
        self.geometry
            .ranks
            .iter()
            .map(|rank| CacheGeometry {
                page_tokens: self.geometry.page_tokens,
                page_capacity: 1,
                row_capacity: 1,
                row: Plane {
                    name: "state".into(),
                    address: 0,
                    unit_bytes: rank.row_bytes,
                },
                planes: rank
                    .plane_bytes
                    .iter()
                    .enumerate()
                    .map(|(index, bytes)| Plane {
                        name: format!("plane{index}"),
                        address: 0,
                        unit_bytes: *bytes,
                    })
                    .collect(),
                snapshot_pages: Vec::new(),
            })
            .collect()
    }
}

impl Geometry {
    /// Canonical bytes: `page_tokens u64`, `page_record_bytes u64`,
    /// `row_record_bytes u64`, `rank count u32`, then per rank
    /// `plane count u32`, each `plane_bytes u64`, and `row_bytes u64`.
    pub fn canonical(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(self.page_tokens.to_le_bytes());
        bytes.extend(self.page_record_bytes.to_le_bytes());
        bytes.extend(self.row_record_bytes.to_le_bytes());
        bytes.extend((self.ranks.len() as u32).to_le_bytes());
        for rank in &self.ranks {
            bytes.extend((rank.plane_bytes.len() as u32).to_le_bytes());
            for plane in &rank.plane_bytes {
                bytes.extend(plane.to_le_bytes());
            }
            bytes.extend(rank.row_bytes.to_le_bytes());
        }
        bytes
    }
}

/// The scope id: BLAKE2b-256 over `"selene-scope-v2"`, the model identity (32
/// bytes), the layout fingerprint (16 bytes), the pool name as one length byte
/// then its bytes, and [`Geometry::canonical`]. Fails for a pool name longer
/// than 255 bytes.
pub fn scope_id(
    pool_name: &str,
    model_identity: &[u8; 32],
    layout_fingerprint: &[u8; 16],
    geometry: &Geometry,
) -> Result<[u8; 32]> {
    let length =
        u8::try_from(pool_name.len()).map_err(|_| Error::new("pool name longer than 255 bytes"))?;
    let mut hash = Blake2b::<U32>::new();
    hash.update(b"selene-scope-v2");
    hash.update(model_identity);
    hash.update(layout_fingerprint);
    hash.update([length]);
    hash.update(pool_name.as_bytes());
    hash.update(geometry.canonical());
    Ok(hash.finalize().into())
}

/// The instance id: BLAKE2b-128 over `"selene-instance-v2"`, then each of the
/// pool_name, the model identity, the layout fingerprint and the GPU UUIDs in
/// ascending byte order, every one preceded by its length as `u32`. Stable
/// across restarts; the order the UUIDs are given in does not matter.
pub fn instance_id(
    pool_name: &str,
    model_identity: &[u8; 32],
    layout_fingerprint: &[u8; 16],
    gpu_uuids: &[[u8; 16]],
) -> [u8; 16] {
    let mut hash = Blake2b::<U16>::new();
    hash.update(b"selene-instance-v2");
    let mut field = |bytes: &[u8]| {
        hash.update((bytes.len() as u32).to_le_bytes());
        hash.update(bytes);
    };
    field(pool_name.as_bytes());
    field(model_identity);
    field(layout_fingerprint);
    let mut uuids = gpu_uuids.to_vec();
    uuids.sort_unstable();
    for uuid in &uuids {
        field(uuid);
    }
    hash.finalize().into()
}

/// 32 fresh random bytes: the secret rank 0 of a multi-process instance gives its peers.
pub fn secret() -> Result<[u8; 32]> {
    use std::io::Read;
    let mut secret = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut secret))
        .map_err(|error| Error::caused_by("cannot draw a rank secret", error))?;
    Ok(secret)
}

/// 16 fresh random bytes naming this process; unchanged across reconnects.
pub fn process_nonce() -> Result<[u8; 16]> {
    use std::io::Read;
    let mut nonce = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut nonce))
        .map_err(|error| Error::caused_by("cannot draw a process nonce", error))?;
    Ok(nonce)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> Geometry {
        Geometry {
            page_tokens: 64,
            ranks: vec![
                RankPlanes {
                    plane_bytes: vec![8192, 4096],
                    row_bytes: 4096,
                };
                2
            ],
            page_record_bytes: 24576,
            row_record_bytes: 8192,
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn the_scope_id_hashes_the_documented_bytes() {
        let id = scope_id("acme", &[1; 32], &[2; 16], &geometry()).unwrap();
        // Computed independently from the layout documented on `scope_id`.
        assert_eq!(
            hex(&id),
            "0028b18b4224d30575d052bfac0e389ae3dd4c24fca7d3866944e0a20159911a"
        );
        assert_ne!(
            id,
            scope_id("acme2", &[1; 32], &[2; 16], &geometry()).unwrap()
        );
        let mut wider = geometry();
        wider.ranks.push(wider.ranks[0].clone());
        assert_ne!(id, scope_id("acme", &[1; 32], &[2; 16], &wider).unwrap());
        assert!(scope_id(&"t".repeat(256), &[1; 32], &[2; 16], &geometry()).is_err());
    }

    #[test]
    fn the_instance_id_is_stable_and_depends_on_every_element() {
        let a = [[1u8; 16], [2u8; 16]];
        let b = [[2u8; 16], [1u8; 16]];
        let id = instance_id("acme", &[1; 32], &[2; 16], &a);
        assert_eq!(id, instance_id("acme", &[1; 32], &[2; 16], &b));
        assert_eq!(hex(&id), "2d1aae5bc9c940dfb6703b820dbb28d8");
        assert_ne!(id, instance_id("acme", &[1; 32], &[2; 16], &a[..1]));
        assert_ne!(id, instance_id("other", &[1; 32], &[2; 16], &a));
        assert_ne!(id, instance_id("acme", &[9; 32], &[2; 16], &a));
        assert_ne!(id, instance_id("acme", &[1; 32], &[9; 16], &a));
        let disjoint = [[3u8; 16], [4u8; 16]];
        assert_ne!(id, instance_id("acme", &[1; 32], &[2; 16], &disjoint));
    }

    #[test]
    fn a_scope_built_from_geometry_matches_the_layout_and_round_trips_through_record_headers() {
        use crate::{geometry::fingerprint, record::Header, testing::geometry};
        let ranks = [
            geometry(64, &[("k", 8192), ("v", 100)], 5000),
            geometry(64, &[("k", 8192), ("v", 100)], 5000),
        ];
        let scope = Scope::new([1; 32], fingerprint(&ranks), &ranks).unwrap();
        let layout = Layout::new(&ranks).unwrap();
        assert_eq!(scope.geometry.page_record_bytes, layout.page_record_bytes);
        assert_eq!(scope.geometry.row_record_bytes, layout.row_record_bytes);
        assert_eq!(scope.geometry.ranks[0].plane_bytes, [8192, 100]);
        // A header built from the scope's sizes describes the same records.
        let key = crate::PrefixKey {
            consumed_tokens: 128,
            digest: [3; 16],
        };
        let header = Header::new(
            &scope.record_geometry(),
            [1; 32],
            scope.layout_fingerprint,
            key,
            None,
            [0; 16],
            2,
            std::time::SystemTime::now(),
        )
        .unwrap();
        assert!(header.matches(&scope.record_geometry()));
        assert_eq!(header.page_record_bytes(), layout.page_record_bytes);
        assert_eq!(header.row_record_bytes(), layout.row_record_bytes);
    }

    #[test]
    fn nonces_differ() {
        assert_ne!(process_nonce().unwrap(), process_nonce().unwrap());
    }
}
