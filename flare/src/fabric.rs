//! TCP endpoints and rank paths on the discovered RoCE interfaces.

use std::net::{Ipv4Addr, SocketAddrV4};

use flare_helios::execution::roce_fabric;
use serde::{Deserialize, Serialize};

pub type Failure = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Link {
    pub from: SocketAddrV4,
    pub to: SocketAddrV4,
}

/// Per-source, per-destination links; non-neighbor entries are empty in a ring.
/// A chain is a ring without the link between the first and last rank. A peer whose topology
/// has no `chain` field reads a chain as direct mode and rejects it as missing links.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Topology {
    pub ring: bool,
    #[serde(default)]
    pub chain: bool,
    pub links: Vec<Vec<Vec<Link>>>,
}

pub fn discover() -> Result<Vec<u8>, Failure> {
    #[cfg(target_os = "linux")]
    {
        let fabric = roce_fabric::discover().map_err(|e| e.to_string())?;
        Ok(roce_fabric::encode(&fabric.endpoints))
    }
    #[cfg(not(target_os = "linux"))]
    Err("fabric transfers require Linux hosts with active RoCE interfaces".into())
}

impl Topology {
    #[cfg(test)]
    pub fn plan(records: &[Vec<u8>], ports: &[u16], ring: bool) -> Result<Self, Failure> {
        Self::plan_shape(records, ports, ring, false)
    }

    /// Plans direct, ring or chain links; at most one of `ring` and `chain` is set.
    pub fn plan_shape(
        records: &[Vec<u8>],
        ports: &[u16],
        ring: bool,
        chain: bool,
    ) -> Result<Self, Failure> {
        if ring && chain {
            return Err("fabric transfers use a ring or a chain, not both".into());
        }
        let world = records.len();
        if !(2..=16).contains(&world) || ports.len() != world || ports.contains(&0) {
            return Err("fabric transfers need 2..16 ranks and their listener ports".into());
        }
        let endpoints = records
            .iter()
            .map(|r| roce_fabric::decode(r))
            .collect::<flare_helios::Result<Vec<_>>>()
            .map_err(|e| e.to_string())?;
        let mut links = vec![vec![Vec::new(); world]; world];
        for from in 0..world {
            for to in from + 1..world {
                if ring && to != from + 1 && !(from == 0 && to == world - 1) {
                    continue;
                }
                if chain && to != from + 1 {
                    continue;
                }
                let pair = vec![endpoints[from].clone(), endpoints[to].clone()];
                // Count reachable rails per edge, rather than all devices on either host.
                let routes = (1..=roce_fabric::MAX_RAILS)
                    .rev()
                    .find_map(|rails| roce_fabric::plan_routes(&pair, 0, rails).ok())
                    .ok_or_else(|| format!("no fabric link from rank {from} to rank {to}"))?;
                for &(local, remote) in &routes[1] {
                    let address = |rank: usize, device: u32| -> Result<SocketAddrV4, Failure> {
                        let ip = endpoints[rank][device as usize]
                            .ipv4
                            .ok_or("fabric TCP requires IPv4 endpoints")?
                            .0;
                        Ok(SocketAddrV4::new(ip, ports[rank]))
                    };
                    let link = Link {
                        from: address(from, local)?,
                        to: address(to, remote)?,
                    };
                    links[to][from].push(Link {
                        from: link.to,
                        to: link.from,
                    });
                    links[from][to].push(link);
                }
            }
        }
        let topology = Self { ring, chain, links };
        topology.validate()?;
        Ok(topology)
    }

    pub fn validate(&self) -> Result<(), Failure> {
        let world = self.links.len();
        if !(2..=16).contains(&world) || self.links.iter().any(|row| row.len() != world) {
            return Err("invalid fabric topology dimensions".into());
        }
        if self.ring && self.chain {
            return Err("fabric topology is both a ring and a chain".into());
        }
        for from in 0..world {
            for to in 0..world {
                let connected = from != to
                    && (!self.ring || to == (from + 1) % world || from == (to + 1) % world);
                let connected = connected && (!self.chain || from.abs_diff(to) == 1);
                let links = &self.links[from][to];
                if connected != !links.is_empty() || links.len() > roce_fabric::MAX_RAILS {
                    return Err("fabric links differ from the selected topology".into());
                }
                if links.len() != self.links[to][from].len() {
                    return Err("fabric links must be symmetric".into());
                }
                for (link, reverse) in links.iter().zip(&self.links[to][from]) {
                    if link.from != reverse.to || link.to != reverse.from {
                        return Err("fabric links must be symmetric".into());
                    }
                    for endpoint in [link.from, link.to] {
                        if endpoint.port() == 0
                            || endpoint.ip().is_unspecified()
                            || endpoint.ip().is_multicast()
                        {
                            return Err("invalid fabric TCP endpoint".into());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn path(&self, source: usize, destination: usize) -> Result<Vec<usize>, Failure> {
        let world = self.links.len();
        if source >= world || destination >= world || source == destination {
            return Err("invalid fabric transfer ranks".into());
        }
        if self.chain {
            // A chain has one route: every rank between the two, in order.
            return Ok(if source < destination {
                (source..=destination).collect()
            } else {
                (destination..=source).rev().collect()
            });
        }
        if !self.ring {
            return Ok(vec![source, destination]);
        }
        let clockwise = (destination + world - source) % world;
        let step = if clockwise <= world - clockwise {
            1
        } else {
            world - 1
        };
        let mut path = vec![source];
        while *path.last().unwrap() != destination {
            path.push((path.last().unwrap() + step) % world);
        }
        Ok(path)
    }

    pub fn hop(
        &self,
        from: usize,
        to: usize,
        rail: usize,
    ) -> Result<(Ipv4Addr, SocketAddrV4), Failure> {
        let links = self
            .links
            .get(from)
            .and_then(|r| r.get(to))
            .filter(|links| !links.is_empty())
            .ok_or("no direct fabric link for this hop")?;
        let link = &links[rail % links.len()];
        Ok((*link.from.ip(), link.to))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_take_shortest_direction_and_clockwise_ties() {
        for world in 2..=16 {
            let records = (1..=world)
                .map(|rank| format!("a 10.0.0.{rank}/24\n").into_bytes())
                .collect::<Vec<_>>();
            let topology = Topology::plan(&records, &vec![29900; world], true).unwrap();
            for source in 0..world {
                for destination in 0..world {
                    if source == destination {
                        assert!(topology.path(source, destination).is_err());
                        continue;
                    }
                    let path = topology.path(source, destination).unwrap();
                    let clockwise = (destination + world - source) % world;
                    assert_eq!(path.len() - 1, clockwise.min(world - clockwise));
                    assert_eq!(path[0], source);
                    assert_eq!(*path.last().unwrap(), destination);
                    for edge in path.windows(2) {
                        assert_eq!(
                            (edge[1] + world - edge[0]) % world,
                            if clockwise <= world - clockwise {
                                1
                            } else {
                                world - 1
                            }
                        );
                    }
                }
            }
            assert!(topology.path(world, 0).is_err());
        }
    }

    #[test]
    fn switched_peers_connect_directly_and_ring_peers_relay() {
        let records = (1..=4)
            .map(|r| format!("a 10.0.0.{r}/24\nb 10.0.1.{r}/24\n").into_bytes())
            .collect::<Vec<_>>();
        for ring in [false, true] {
            let topology = Topology::plan(&records, &[29900; 4], ring).unwrap();
            assert_eq!(
                topology.path(0, 2).unwrap(),
                if ring { vec![0, 1, 2] } else { vec![0, 2] }
            );
            assert_eq!(topology.hop(0, 2, 0).is_err(), ring);
            assert_eq!(topology.path(0, 3).unwrap(), [0, 3]);
        }
    }

    #[test]
    fn physical_ring_counts_rails_per_neighbor() {
        let records = [
            "a 10.0.0.1/24\nb 10.0.3.2/24\n",
            "a 10.0.1.1/24\nb 10.0.0.2/24\n",
            "a 10.0.2.1/24\nb 10.0.1.2/24\n",
            "a 10.0.3.1/24\nb 10.0.2.2/24\n",
        ]
        .map(|s| s.as_bytes().to_vec());
        let topology = Topology::plan(&records, &[29900; 4], true).unwrap();
        assert_eq!(topology.links[0][1].len(), 1);
        assert_eq!(topology.path(0, 2).unwrap(), [0, 1, 2]);
        assert!(Topology::plan(&records, &[29900; 4], false).is_err());
    }

    #[test]
    fn chain_paths_walk_every_rank_between_the_ends() {
        for world in 2..=16 {
            let records = (1..=world)
                .map(|rank| format!("a 10.0.0.{rank}/24\n").into_bytes())
                .collect::<Vec<_>>();
            let topology =
                Topology::plan_shape(&records, &vec![29900; world], false, true).unwrap();
            assert_eq!(topology.links[0][world - 1].is_empty(), world > 2);
            for source in 0..world {
                for destination in 0..world {
                    if source == destination {
                        assert!(topology.path(source, destination).is_err());
                        continue;
                    }
                    let path = topology.path(source, destination).unwrap();
                    assert_eq!(path.len() - 1, source.abs_diff(destination));
                    assert_eq!((path[0], *path.last().unwrap()), (source, destination));
                    for edge in path.windows(2) {
                        assert!(topology.hop(edge[0], edge[1], 0).is_ok());
                    }
                }
            }
        }
    }

    #[test]
    fn physical_chain_plans_without_a_closing_link() {
        // Ranks 0 and 3 share no subnet: a ring cannot close, a chain does not need to.
        let records = [
            "a 10.0.0.1/24\n",
            "a 10.0.1.1/24\nb 10.0.0.2/24\n",
            "a 10.0.2.1/24\nb 10.0.1.2/24\n",
            "b 10.0.2.2/24\n",
        ]
        .map(|s| s.as_bytes().to_vec());
        let topology = Topology::plan_shape(&records, &[29900; 4], false, true).unwrap();
        assert_eq!(topology.path(3, 0).unwrap(), [3, 2, 1, 0]);
        assert!(topology.hop(0, 3, 0).is_err());
        assert!(Topology::plan(&records, &[29900; 4], true).is_err());
        assert!(Topology::plan_shape(&records, &[29900; 4], true, true).is_err());
        let mut both = topology.clone();
        both.ring = true;
        assert!(both.validate().is_err());
        let mut direct = topology;
        direct.chain = false;
        assert!(direct.validate().is_err());
    }
}
