//! Orbit authorization around the shared fabric transfer routes.

use super::{Failure, grant::Grant, store::Node};
use crate::fabric::Topology;
pub use crate::fabric::discover;

pub fn plan(
    nodes: &[Node],
    records: &[Vec<u8>],
    ring: bool,
    chain: bool,
) -> Result<Topology, Failure> {
    let ports = nodes
        .iter()
        .map(|node| {
            super::with_port(&node.address)
                .rsplit_once(':')
                .ok_or("agent address lacks a port")?
                .1
                .parse::<u16>()
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>, Failure>>()?;
    Topology::plan_shape(records, &ports, ring, chain)
}

pub fn topology(grant: &Grant) -> Result<&Topology, Failure> {
    let topology = grant
        .transfer_fabric
        .as_ref()
        .ok_or("transfer lacks signed fabric endpoints")?;
    topology.validate()?;
    if topology.links.len() != grant.members.len()
        || topology.ring != grant.roce_ring
        || topology.chain != grant.roce_chain
    {
        return Err("fabric topology differs from the launch membership".into());
    }
    for (rank, member) in grant.members.iter().enumerate() {
        if member.rank != rank
            || grant.members[..rank]
                .iter()
                .any(|m| m.public_key == member.public_key)
        {
            return Err("fabric members must have distinct keys in rank order".into());
        }
    }
    Ok(topology)
}

/// An outside package source stages its bytes on rank zero before fabric distribution.
pub fn source_rank(grant: &Grant) -> usize {
    grant
        .package
        .source
        .as_ref()
        .and_then(|source| {
            grant
                .members
                .iter()
                .position(|member| member.public_key == source.public_key)
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_ring_selects_only_neighbor_subnets_on_both_rails() {
        let nodes = (0..4)
            .map(|rank| Node {
                name: format!("rank{rank}"),
                fingerprint: String::new(),
                public_key: rank.to_string(),
                address: format!("10.137.203.{}:29900", rank + 1),
            })
            .collect::<Vec<_>>();
        let records = [
            "a 10.10.4.1/24\nb 10.10.0.2/24\nc 10.10.5.1/24\nd 10.10.1.2/24\n",
            "a 10.10.6.1/24\nb 10.10.4.2/24\nc 10.10.7.1/24\nd 10.10.5.2/24\n",
            "a 10.10.2.2/24\nb 10.10.6.2/24\nc 10.10.3.2/24\nd 10.10.7.2/24\n",
            "a 10.10.0.1/24\nb 10.10.2.1/24\nc 10.10.1.1/24\nd 10.10.3.1/24\n",
        ]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect::<Vec<_>>();
        let ring = plan(&nodes, &records, true, false).unwrap();
        for (rank, subnets) in [[4, 5], [6, 7], [2, 3], [0, 1]].into_iter().enumerate() {
            let edge = &ring.links[rank][(rank + 1) % 4];
            assert_eq!(edge.len(), 2);
            for (link, subnet) in edge.iter().zip(subnets) {
                assert_eq!(link.from.ip().octets()[..3], [10, 10, subnet]);
                assert_eq!(link.to.ip().octets()[..3], [10, 10, subnet]);
                assert_ne!(link.from.ip(), link.to.ip());
            }
        }
        let mut broken = records.clone();
        broken[2] = b"a 192.168.0.1/24\n".to_vec();
        assert!(plan(&nodes, &broken, true, false).is_err());
    }
}
