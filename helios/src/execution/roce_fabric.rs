//! The RoCE fabric a rank's transport uses, found on its host when it serves.
//!
//! A package records what its kernels need of the transport (world, region layout, proxy ABI),
//! but not the fabric it was prepared on: which RDMA devices a host has, their GIDs, and how
//! hosts are cabled differ between clusters and even between launches. Each rank discovers its
//! active RoCEv2 devices, ranks exchange those endpoint lists at the rendezvous, and every rank
//! plans the same routes from them: which local and remote device carries each rail to each peer.
//!
//! Routing follows b12x's `_routes.py`. On a switched fabric rail `h` pairs device `h` at both
//! ends. Hosts cabled point to point (three Sparks in a ring) reach a neighbour only through
//! devices on a shared subnet, so rails then pair devices per peer by subnet, ordered by link
//! address so both ends agree.

use std::net::Ipv4Addr;
use std::path::Path;

use crate::{ExecutionError, Failure, Result};

/// Each peer is reached over at most this many devices.
pub const MAX_RAILS: usize = 2;
/// A rank opens at most this many devices.
pub const MAX_DEVICES: usize = 4;
/// b12x's default GID index: RoCEv2 over IPv4 on a DGX Spark.
const PREFERRED_GID_INDEX: u32 = 3;
const GID_TABLE: u32 = 16;

fn failure(message: impl Into<String>) -> crate::Error {
    ExecutionError::Poisoned(Failure::new(message.into())).into()
}

/// One RDMA device: its name and, when its GID is IPv4-mapped, the address and prefix length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub name: String,
    pub ipv4: Option<(Ipv4Addr, u8)>,
}

impl Endpoint {
    fn network(&self) -> Option<(u32, u8)> {
        self.ipv4.map(|(address, prefix)| {
            let mask = if prefix == 0 {
                0
            } else {
                u32::MAX << (32 - prefix as u32)
            };
            (u32::from(address) & mask, prefix)
        })
    }
}

/// This host's RoCE devices in name order, and the GID index and traffic class they use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fabric {
    pub gid_index: u32,
    pub traffic_class: u32,
    pub endpoints: Vec<Endpoint>,
}

/// How the ranks of a multi-host package link over RoCE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// Every rank links to every other rank.
    Mesh,
    /// Experimental ring in rank order: each rank links to its two neighbors and the native proxy
    /// forwards writes to every other rank around the shorter arc.
    Ring,
    /// Experimental chain in rank order: a ring with no link between rank 0 and the last rank, so
    /// the ends have one neighbor and the proxy forwards along the chain.
    Chain,
}

impl Shape {
    /// The shape two command-line flags select; the chain flag wins because it implies relaying.
    pub fn from_flags(ring: bool, chain: bool) -> Self {
        if chain {
            Shape::Chain
        } else if ring {
            Shape::Ring
        } else {
            Shape::Mesh
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Shape::Mesh => "mesh",
            Shape::Ring => "ring",
            Shape::Chain => "chain",
        }
    }

    /// Whether the native proxy relays writes between ranks that hold no direct link.
    pub fn relays(self) -> bool {
        self != Shape::Mesh
    }

    /// Whether `rank` holds a direct link to `peer` among `world` ranks; a rank never links to itself.
    pub fn linked(self, rank: usize, peer: usize, world: usize) -> bool {
        match self {
            Shape::Mesh => peer != rank,
            Shape::Ring => {
                peer != rank && (peer == (rank + 1) % world || peer == (rank + world - 1) % world)
            }
            Shape::Chain => peer + 1 == rank || peer == rank + 1,
        }
    }
}

/// The devices this rank opens and, per peer, the (local, remote) device indexes of each rail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    pub hcas: Vec<String>,
    pub gid_index: u32,
    pub traffic_class: u32,
    pub rails: usize,
    pub routes: Vec<Vec<(u32, u32)>>,
    pub shape: Shape,
}

fn read(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|text| text.trim().to_owned())
}

/// The IPv4 address an IPv4-mapped GID (::ffff:a.b.c.d) holds.
fn mapped_ipv4(gid: &str) -> Option<Ipv4Addr> {
    let hex: String = gid.chars().filter(|c| *c != ':').collect();
    if hex.len() != 32
        || !hex[..20].bytes().all(|b| b == b'0')
        || !hex[20..24].eq_ignore_ascii_case("ffff")
    {
        return None;
    }
    u32::from_str_radix(&hex[24..], 16).ok().map(Ipv4Addr::from)
}

/// Discover the RoCE devices under `root` (normally /sys/class/infiniband): every device whose
/// port 1 is active and holds a RoCEv2 GID at the chosen index. The index is 3 when every active
/// device has a RoCEv2 IPv4 GID there, otherwise the lowest index at which one does. `prefix`
/// gives a network device's prefix length for an address it holds.
pub fn discover_in(root: &Path, prefix: impl Fn(&str, Ipv4Addr) -> Option<u8>) -> Result<Fabric> {
    let mut names: Vec<String> = std::fs::read_dir(root)
        .map_err(|error| {
            failure(format!(
                "cannot list RDMA devices in {}: {error}",
                root.display()
            ))
        })?
        .filter_map(|entry| {
            entry
                .ok()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
        })
        .filter(|name| {
            read(&root.join(name).join("ports/1/state"))
                .is_some_and(|state| state.contains("ACTIVE"))
        })
        .collect();
    names.sort();
    let roce_v2_ipv4 = |name: &str, index: u32| {
        let port = root.join(name).join("ports/1");
        read(&port.join(format!("gid_attrs/types/{index}")))
            .is_some_and(|kind| kind.eq_ignore_ascii_case("RoCE v2"))
            && read(&port.join(format!("gids/{index}")))
                .as_deref()
                .and_then(mapped_ipv4)
                .is_some()
    };
    let gid_index = if !names.is_empty()
        && names
            .iter()
            .all(|name| roce_v2_ipv4(name, PREFERRED_GID_INDEX))
    {
        PREFERRED_GID_INDEX
    } else {
        (0..GID_TABLE)
            .find(|&index| names.iter().any(|name| roce_v2_ipv4(name, index)))
            .ok_or_else(|| failure("no active RDMA device has a RoCEv2 IPv4 GID"))?
    };
    let endpoints: Vec<Endpoint> = names
        .iter()
        .filter(|name| roce_v2_ipv4(name, gid_index))
        .take(MAX_DEVICES)
        .map(|name| {
            let port = root.join(name).join("ports/1");
            let address = read(&port.join(format!("gids/{gid_index}")))
                .as_deref()
                .and_then(mapped_ipv4);
            let device = read(&port.join(format!("gid_attrs/ndevs/{gid_index}")));
            let ipv4 = address
                .zip(device)
                .and_then(|(address, device)| prefix(&device, address).map(|p| (address, p)));
            Endpoint {
                name: name.clone(),
                ipv4,
            }
        })
        .collect();
    Ok(Fabric {
        gid_index,
        traffic_class: 0,
        endpoints,
    })
}

/// The prefix length of the IPv4 address `address` on network device `device`, from getifaddrs.
#[cfg(target_os = "linux")]
fn interface_prefix(device: &str, address: Ipv4Addr) -> Option<u8> {
    let mut list: *mut libc::ifaddrs = std::ptr::null_mut();
    // SAFETY: getifaddrs fills a list this function frees before returning; it reads entries only
    // while the list is live and checks every pointer it dereferences.
    unsafe {
        if libc::getifaddrs(&mut list) != 0 {
            return None;
        }
        let mut found = None;
        let mut entry = list;
        while !entry.is_null() {
            let item = &*entry;
            entry = item.ifa_next;
            if item.ifa_addr.is_null()
                || item.ifa_netmask.is_null()
                || (*item.ifa_addr).sa_family as i32 != libc::AF_INET
            {
                continue;
            }
            if std::ffi::CStr::from_ptr(item.ifa_name).to_str() != Ok(device) {
                continue;
            }
            let held = &*(item.ifa_addr as *const libc::sockaddr_in);
            if Ipv4Addr::from(u32::from_be(held.sin_addr.s_addr)) != address {
                continue;
            }
            let mask = &*(item.ifa_netmask as *const libc::sockaddr_in);
            found = Some(u32::from_be(mask.sin_addr.s_addr).count_ones() as u8);
            break;
        }
        libc::freeifaddrs(list);
        found
    }
}

/// Discover this host's RoCE devices.
#[cfg(target_os = "linux")]
pub fn discover() -> Result<Fabric> {
    discover_in(Path::new("/sys/class/infiniband"), interface_prefix)
}

/// Encode endpoints for the rendezvous exchange: one `name address/prefix` line per device, or
/// `name -` for a device without an IPv4 GID.
pub fn encode(endpoints: &[Endpoint]) -> Vec<u8> {
    endpoints
        .iter()
        .map(|e| match e.ipv4 {
            Some((address, prefix)) => format!("{} {address}/{prefix}\n", e.name),
            None => format!("{} -\n", e.name),
        })
        .collect::<String>()
        .into_bytes()
}

pub fn decode(bytes: &[u8]) -> Result<Vec<Endpoint>> {
    let text = std::str::from_utf8(bytes).map_err(|_| failure("RoCE endpoints are not text"))?;
    text.lines()
        .map(|line| {
            let (name, address) = line
                .split_once(' ')
                .ok_or_else(|| failure("malformed RoCE endpoint"))?;
            let ipv4 = match address {
                "-" => None,
                _ => {
                    let (ip, prefix) = address
                        .split_once('/')
                        .ok_or_else(|| failure("malformed RoCE endpoint"))?;
                    let ip = ip
                        .parse()
                        .map_err(|_| failure("malformed RoCE endpoint address"))?;
                    let prefix: u8 = prefix
                        .parse()
                        .map_err(|_| failure("malformed RoCE endpoint prefix"))?;
                    if prefix > 32 {
                        return Err(failure("malformed RoCE endpoint prefix"));
                    }
                    Some((ip, prefix))
                }
            };
            Ok(Endpoint {
                name: name.to_owned(),
                ipv4,
            })
        })
        .collect()
}

fn same_link(a: &Endpoint, b: &Endpoint) -> bool {
    matches!((a.network(), b.network()), (Some(x), Some(y)) if x == y)
}

/// True when rail `h` can pair device `h` with device `h` between every pair of ranks.
fn index_pairing_valid(endpoints: &[Vec<Endpoint>], rails: usize) -> bool {
    if endpoints.iter().any(|devices| devices.len() < rails) {
        return false;
    }
    for a in 0..endpoints.len() {
        for b in a + 1..endpoints.len() {
            for h in 0..rails {
                let (x, y) = (&endpoints[a][h], &endpoints[b][h]);
                // A device without an IPv4 GID cannot be checked: trust the device order.
                if x.ipv4.is_some() && y.ipv4.is_some() && !same_link(x, y) {
                    return false;
                }
            }
        }
    }
    true
}

type Key = (u32, u64, u64, u64);

/// The `rails` candidates with distinct local and remote devices that use the most verified
/// links, ties broken by the smallest sorted keys; None when no such set exists.
fn best_matching(candidates: &[(Key, u32, u32)], rails: usize) -> Option<Vec<(u32, u32)>> {
    let mut best: Option<((u32, Vec<Key>), Vec<(u32, u32)>)> = None;
    let mut visit = |combo: &[usize]| {
        let chosen: Vec<_> = combo.iter().map(|&i| candidates[i]).collect();
        let locals: std::collections::BTreeSet<_> = chosen.iter().map(|c| c.1).collect();
        let remotes: std::collections::BTreeSet<_> = chosen.iter().map(|c| c.2).collect();
        if locals.len() < rails || remotes.len() < rails {
            return;
        }
        let score = (
            chosen.iter().map(|c| c.0.0).sum(),
            chosen.iter().map(|c| c.0).collect(),
        );
        if best.as_ref().is_none_or(|(current, _)| score < *current) {
            best = Some((score, chosen.iter().map(|c| (c.1, c.2)).collect()));
        }
    };
    // Combinations in lexicographic order, as itertools.combinations yields them.
    let n = candidates.len();
    if rails == 0 || rails > n {
        return None;
    }
    let mut index: Vec<usize> = (0..rails).collect();
    loop {
        visit(&index);
        let Some(i) = (0..rails).rev().find(|&i| index[i] != i + n - rails) else {
            break;
        };
        index[i] += 1;
        for j in i + 1..rails {
            index[j] = index[j - 1] + 1;
        }
    }
    best.map(|(_, links)| links)
}

/// Per peer, the (local device, remote device) indexes carrying each rail; empty for `rank`.
pub fn plan_routes(
    endpoints: &[Vec<Endpoint>],
    rank: usize,
    rails: usize,
) -> Result<Vec<Vec<(u32, u32)>>> {
    if !(1..=MAX_RAILS).contains(&rails) || rank >= endpoints.len() {
        return Err(failure("invalid RoCE route request"));
    }
    if index_pairing_valid(endpoints, rails) {
        return Ok((0..endpoints.len())
            .map(|peer| {
                if peer == rank {
                    Vec::new()
                } else {
                    (0..rails as u32).map(|h| (h, h)).collect()
                }
            })
            .collect());
    }
    let local = &endpoints[rank];
    let mut routes = Vec::with_capacity(endpoints.len());
    for (peer, remote) in endpoints.iter().enumerate() {
        if peer == rank {
            routes.push(Vec::new());
            continue;
        }
        // Verified same-subnet links keyed by network and the link's two addresses, and same-index
        // pairs without an IPv4 GID at either end, trusted in device order. Keys are symmetric.
        let mut candidates: Vec<(Key, u32, u32)> = Vec::new();
        for (l, x) in local.iter().enumerate() {
            for (r, y) in remote.iter().enumerate() {
                if same_link(x, y) {
                    let (a, b) = (
                        u32::from(x.ipv4.unwrap().0) as u64,
                        u32::from(y.ipv4.unwrap().0) as u64,
                    );
                    candidates.push((
                        (0, x.network().unwrap().0 as u64, a.min(b), a.max(b)),
                        l as u32,
                        r as u32,
                    ));
                }
            }
        }
        for h in 0..local.len().min(remote.len()) {
            if local[h].ipv4.is_none() || remote[h].ipv4.is_none() {
                candidates.push(((1, h as u64, 0, 0), h as u32, h as u32));
            }
        }
        candidates.sort();
        let chosen = best_matching(&candidates, rails).ok_or_else(|| {
            let reach = (1..rails).rev().find(|&k| best_matching(&candidates, k).is_some()).unwrap_or(0);
            failure(format!(
                "rank {rank}: {reach} of {rails} RoCE rails have a link to rank {peer}; local {local:?}, remote {remote:?}"
            ))
        })?;
        routes.push(chosen);
    }
    Ok(routes)
}

/// The routes of `shape` for `rank`: a mesh routes to every rank; a ring or chain routes only to
/// the ranks `shape` links, and the native proxy forwards writes to the others. The same plan works
/// over a switch while keeping unlinked QPs unconnected.
pub fn plan_shape_routes(
    endpoints: &[Vec<Endpoint>],
    rank: usize,
    rails: usize,
    shape: Shape,
) -> Result<Vec<Vec<(u32, u32)>>> {
    if shape == Shape::Mesh {
        return plan_routes(endpoints, rank, rails);
    }
    if !(2..=16).contains(&endpoints.len()) || rank >= endpoints.len() {
        return Err(failure("experimental RoCE ring requires 2..16 ranks"));
    }
    let world = endpoints.len();
    let peers: Vec<_> = (0..world).filter(|p| *p == rank || shape.linked(rank, *p, world)).collect();
    let subset: Vec<_> = peers.iter().map(|p| endpoints[*p].clone()).collect();
    let local = peers.iter().position(|p| *p == rank).unwrap();
    let planned = plan_routes(&subset, local, rails)?;
    Ok((0..world)
        .map(|p| {
            peers
                .iter()
                .position(|q| *q == p)
                .map_or_else(Vec::new, |i| planned[i].clone())
        })
        .collect())
}

impl Fabric {
    /// The topology this rank opens for `shape`, from every rank's endpoints in rank order.
    pub fn topology(&self, endpoints: &[Vec<Endpoint>], rank: usize, shape: Shape) -> Result<Topology> {
        if endpoints.get(rank) != Some(&self.endpoints) || endpoints.iter().any(Vec::is_empty) {
            return Err(failure("every rank needs at least one RoCE device"));
        }
        let rails = endpoints.iter().map(Vec::len).min().unwrap().min(MAX_RAILS);
        Ok(Topology {
            hcas: self.endpoints.iter().map(|e| e.name.clone()).collect(),
            gid_index: self.gid_index,
            traffic_class: self.traffic_class,
            rails,
            routes: plan_shape_routes(endpoints, rank, rails, shape)?,
            shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(name: &str, address: &str) -> Endpoint {
        let (ip, prefix) = address.split_once('/').unwrap();
        Endpoint {
            name: name.into(),
            ipv4: Some((ip.parse().unwrap(), prefix.parse().unwrap())),
        }
    }

    fn gid(address: Ipv4Addr) -> String {
        let [a, b, c, d] = address.octets();
        format!("0000:0000:0000:0000:0000:ffff:{a:02x}{b:02x}:{c:02x}{d:02x}")
    }

    fn device(root: &Path, name: &str, state: &str, gids: &[(u32, &str, Ipv4Addr)]) {
        let port = root.join(name).join("ports/1");
        for directory in ["gids", "gid_attrs/types", "gid_attrs/ndevs"] {
            std::fs::create_dir_all(port.join(directory)).unwrap();
        }
        std::fs::write(port.join("state"), format!("{state}\n")).unwrap();
        for (index, kind, address) in gids {
            std::fs::write(port.join(format!("gids/{index}")), gid(*address)).unwrap();
            std::fs::write(
                port.join(format!("gid_attrs/types/{index}")),
                format!("{kind}\n"),
            )
            .unwrap();
            std::fs::write(
                port.join(format!("gid_attrs/ndevs/{index}")),
                format!("net-{name}\n"),
            )
            .unwrap();
        }
    }

    fn scratch(label: &str) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("helios-fabric-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn active_roce_v2_devices_are_found_in_name_order_with_their_prefixes() {
        let root = scratch("found");
        let (a, b) = (Ipv4Addr::new(10, 200, 1, 12), Ipv4Addr::new(10, 200, 0, 12));
        device(
            &root,
            "rocep1s0f0",
            "4: ACTIVE",
            &[(1, "IB/RoCE v1", a), (3, "RoCE v2", a)],
        );
        device(&root, "roceP2p1s0f0", "4: ACTIVE", &[(3, "RoCE v2", b)]);
        device(
            &root,
            "rocep1s0f1",
            "1: DOWN",
            &[(3, "RoCE v2", Ipv4Addr::new(10, 9, 9, 9))],
        );
        let fabric = discover_in(&root, |device, _| {
            (device != "net-rocep1s0f0").then_some(24)
        })
        .unwrap();
        assert_eq!(fabric.gid_index, 3);
        assert_eq!(
            fabric.endpoints,
            vec![
                Endpoint {
                    name: "roceP2p1s0f0".into(),
                    ipv4: Some((b, 24))
                },
                // A device whose netmask cannot be read has no checkable address.
                Endpoint {
                    name: "rocep1s0f0".into(),
                    ipv4: None
                },
            ]
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn another_roce_v2_index_serves_when_index_three_is_not_one() {
        let root = scratch("index");
        let a = Ipv4Addr::new(192, 168, 7, 2);
        device(
            &root,
            "mlx5_0",
            "4: ACTIVE",
            &[(1, "RoCE v2", a), (3, "IB/RoCE v1", a)],
        );
        let fabric = discover_in(&root, |_, _| Some(16)).unwrap();
        assert_eq!((fabric.gid_index, fabric.endpoints.len()), (1, 1));
        std::fs::remove_dir_all(&root).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        assert!(discover_in(&root, |_, _| Some(16)).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn endpoints_survive_the_rendezvous_encoding() {
        let endpoints = vec![
            endpoint("rocep1s0f0", "10.200.1.11/24"),
            Endpoint {
                name: "x".into(),
                ipv4: None,
            },
        ];
        assert_eq!(decode(&encode(&endpoints)).unwrap(), endpoints);
        assert!(decode(b"rocep1s0f0 10.0.0.1/40\n").is_err());
    }

    #[test]
    fn a_switched_fabric_pairs_devices_by_index() {
        let rank = |host: u8| {
            vec![
                endpoint("f0", &format!("10.200.1.{host}/24")),
                endpoint("f1", &format!("10.200.0.{host}/24")),
            ]
        };
        let endpoints = vec![rank(12), rank(13), rank(11), rank(14)];
        let routes = plan_routes(&endpoints, 1, 2).unwrap();
        assert_eq!(
            routes,
            vec![
                vec![(0, 0), (1, 1)],
                vec![],
                vec![(0, 0), (1, 1)],
                vec![(0, 0), (1, 1)]
            ]
        );
    }

    #[test]
    fn a_ring_of_sparks_pairs_devices_by_subnet_and_both_ends_agree() {
        // Each Spark's port 0 is cabled to the next Spark's port 1; each port is two functions on
        // their own point-to-point subnets.
        // The subnet between Sparks a and b (port 0 of a, port 1 of b) is 10.<link>.<function>.0/30.
        let link = |a: u32, b: u32| 10 * (a + 1) + b + 1;
        let spark = |me: u32| {
            let next = (me + 1) % 3;
            let previous = (me + 2) % 3;
            vec![
                endpoint("p0f0", &format!("10.{}.0.1/30", link(me, next))),
                endpoint("p0f1", &format!("10.{}.1.1/30", link(me, next))),
                endpoint("p1f0", &format!("10.{}.0.2/30", link(previous, me))),
                endpoint("p1f1", &format!("10.{}.1.2/30", link(previous, me))),
            ]
        };
        let endpoints: Vec<_> = (0..3).map(spark).collect();
        let plans: Vec<_> = (0..3)
            .map(|rank| plan_routes(&endpoints, rank, 2).unwrap())
            .collect();
        for a in 0..3 {
            for b in 0..3 {
                if a == b {
                    continue;
                }
                let mirrored: Vec<_> = plans[b][a].iter().map(|&(l, r)| (r, l)).collect();
                assert_eq!(plans[a][b], mirrored, "ranks {a} and {b} disagree");
                for &(l, r) in &plans[a][b] {
                    assert!(same_link(
                        &endpoints[a][l as usize],
                        &endpoints[b][r as usize]
                    ));
                }
            }
        }
        // Rank 0 reaches rank 1 over its port 0 and rank 2 over its port 1.
        assert_eq!(plans[0][1], vec![(0, 2), (1, 3)]);
        assert_eq!(plans[0][2], vec![(2, 0), (3, 1)]);
        // With one function per port, only one rail reaches each neighbour.
        let single: Vec<Vec<Endpoint>> = endpoints
            .iter()
            .map(|e| vec![e[0].clone(), e[2].clone()])
            .collect();
        let error = plan_routes(&single, 0, 2).unwrap_err().to_string();
        assert!(error.contains("1 of 2 RoCE rails"), "{error}");
    }

    #[test]
    fn variable_size_rings_route_only_neighbors_on_physical_and_switched_fabrics() {
        for world in [2, 3, 4, 5, 6, 8, 16] {
            for switched in [false, true] {
                let endpoints: Vec<Vec<Endpoint>> = (0..world)
                    .map(|rank| {
                        if switched {
                            return vec![
                                endpoint("f0", &format!("10.1.0.{}/24", rank + 1)),
                                endpoint("f1", &format!("10.2.0.{}/24", rank + 1)),
                            ];
                        }
                        let previous = (rank + world - 1) % world;
                        vec![
                            endpoint("p0f0", &format!("10.{}.0.1/30", rank + 1)),
                            endpoint("p0f1", &format!("10.{}.1.1/30", rank + 1)),
                            endpoint("p1f0", &format!("10.{}.0.2/30", previous + 1)),
                            endpoint("p1f1", &format!("10.{}.1.2/30", previous + 1)),
                        ]
                    })
                    .collect();
                if !switched && world > 3 {
                    assert!(plan_routes(&endpoints, 0, 2).is_err());
                }
                let plans: Vec<_> = (0..world)
                    .map(|r| plan_shape_routes(&endpoints, r, 2, Shape::Ring).unwrap())
                    .collect();
                for rank in 0..world {
                    assert!(plans[rank][rank].is_empty());
                    for peer in 0..world {
                        if peer == rank
                            || peer == (rank + 1) % world
                            || peer == (rank + world - 1) % world
                        {
                            continue;
                        }
                        assert!(plans[rank][peer].is_empty());
                    }
                    for peer in [(rank + 1) % world, (rank + world - 1) % world] {
                        assert_eq!(plans[rank][peer].len(), 2);
                        let reverse: Vec<_> =
                            plans[peer][rank].iter().map(|&(r, l)| (l, r)).collect();
                        assert_eq!(plans[rank][peer], reverse);
                    }
                }
            }
        }
    }

    #[test]
    fn ring_rejects_wrong_world_size_and_missing_neighbor_links() {
        assert!(plan_shape_routes(&[], 0, 2, Shape::Ring).is_err());
        let endpoints = vec![vec![endpoint("f0", "10.0.0.1/30")]; 4];
        assert!(plan_shape_routes(&endpoints, 4, 2, Shape::Ring).is_err());
        assert!(plan_shape_routes(&endpoints, 0, 2, Shape::Ring).is_err());
    }

    #[test]
    fn a_line_connects_neighbors_only_and_needs_no_link_between_its_ends() {
        // Four hosts cabled 0-1, 1-2, 2-3 on distinct /30 subnets; the ends share no subnet.
        let endpoints = vec![
            vec![endpoint("a", "10.0.0.1/30")],
            vec![endpoint("b", "10.0.0.2/30"), endpoint("c", "10.0.1.1/30")],
            vec![endpoint("d", "10.0.1.2/30"), endpoint("e", "10.0.2.1/30")],
            vec![endpoint("f", "10.0.2.2/30")],
        ];
        let routes = plan_shape_routes(&endpoints, 0, 1, Shape::Chain).unwrap();
        assert_eq!(routes, vec![vec![], vec![(0, 0)], vec![], vec![]]);
        let routes = plan_shape_routes(&endpoints, 3, 1, Shape::Chain).unwrap();
        assert_eq!(routes, vec![vec![], vec![], vec![(0, 1)], vec![]]);
        let routes = plan_shape_routes(&endpoints, 1, 1, Shape::Chain).unwrap();
        assert_eq!(routes, vec![vec![(0, 0)], vec![], vec![(1, 0)], vec![]]);
        // The same hosts as a ring need the missing 3-0 link.
        assert!(plan_shape_routes(&endpoints, 0, 1, Shape::Ring).is_err());
    }

    #[test]
    fn a_topology_opens_every_device_and_caps_rails_at_the_smallest_rank() {
        let fabric = Fabric {
            gid_index: 3,
            traffic_class: 0,
            endpoints: vec![endpoint("a", "10.0.0.1/24"), endpoint("b", "10.0.1.1/24")],
        };
        let peer = vec![endpoint("c", "10.0.0.2/24")];
        let topology = fabric
            .topology(&[fabric.endpoints.clone(), peer], 0, Shape::Mesh)
            .unwrap();
        assert_eq!(
            (
                topology.hcas.len(),
                topology.rails,
                topology.routes[1].clone()
            ),
            (2, 1, vec![(0, 0)])
        );
        assert!(
            fabric
                .topology(&[fabric.endpoints.clone(), vec![]], 0, Shape::Mesh)
                .is_err()
        );
    }
}
