//! Files that hold orbit state: the agent's directory on a node and the controller's orbit file.
//!
//! Node side, under `~/.config/flare/agent/`: `node.key` (the identity), `controllers.toml`
//! (the signed controller-set record; absent on an unpaired node), `join-code` (the SHA-256 of
//! an active join code and its expiry), and `accept-first` (a marker admitting the first
//! controller without a code).
//!
//! Controller side: `~/.config/flare/orbit/controller.key` and `~/.config/flare/orbit.toml`,
//! which lists the controller set, the paired nodes, and named clusters in rank order.

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::grant::{ControllerSet, Signed, verify_controller_set};
use super::identity::{Key, PublicKey};
use super::{Failure, hex, home, now, sha256};

// ---- agent -------------------------------------------------------------------------------------

pub struct AgentDir {
    pub root: PathBuf,
}

/// An active join code: the HMAC key both sides derive from it, and when it stops working.
pub struct JoinCode {
    pub key: [u8; 32],
    pub expires: u64,
}

impl AgentDir {
    pub fn default_root() -> PathBuf {
        home().join(".config/flare/agent")
    }

    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn key(&self) -> Result<Key, Failure> {
        Key::load_or_create(&self.root.join("node.key"))
    }

    fn private_write(&self, path: &Path, text: &str) -> Result<(), Failure> {
        fs::create_dir_all(&self.root)?;
        fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))?;
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, text)?;
        fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600))?;
        fs::rename(&temporary, path)?;
        Ok(())
    }

    /// The stored controller-set record and its verified set, or None on an unpaired node.
    pub fn controllers(&self) -> Result<Option<(Signed, ControllerSet)>, Failure> {
        let path = self.root.join("controllers.toml");
        match fs::read_to_string(&path) {
            Ok(text) => {
                let record: Signed = toml::from_str(&text)
                    .map_err(|e| format!("invalid {}: {e}", path.display()))?;
                // The stored record was verified when written; its signer is its own witness.
                let set: ControllerSet = record.verify(|_| true)?;
                Ok(Some((record, set)))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!("cannot read {}: {error}", path.display()).into()),
        }
    }

    /// Apply a controller-set record: the first on an unpaired node, or a later revision signed
    /// by a current member.
    pub fn apply_controllers(&self, record: &Signed) -> Result<ControllerSet, Failure> {
        let stored = self.controllers()?;
        let set = verify_controller_set(record, stored.as_ref().map(|(_, set)| set))?;
        self.private_write(
            &self.root.join("controllers.toml"),
            &toml::to_string(record)?,
        )?;
        Ok(set)
    }

    pub fn reset(&self) -> Result<(), Failure> {
        for name in ["controllers.toml", "join-code", "accept-first"] {
            match fs::remove_file(self.root.join(name)) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    /// Install `code` (as typed) as the active join code for `seconds`.
    pub fn set_join_code(&self, code: &str, seconds: u64) -> Result<(), Failure> {
        let key = join_key(code);
        self.private_write(
            &self.root.join("join-code"),
            &format!("{} {}\n", hex(&key), now() + seconds),
        )
    }

    pub fn join_code(&self) -> Result<Option<JoinCode>, Failure> {
        let path = self.root.join("join-code");
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let mut parts = text.split_whitespace();
        let (Some(key), Some(expires)) = (parts.next(), parts.next()) else {
            return Err(format!("invalid join code file {}", path.display()).into());
        };
        let code = JoinCode {
            key: super::unhex32(key)?,
            expires: expires.parse()?,
        };
        if now() > code.expires {
            let _ = fs::remove_file(&path);
            return Ok(None);
        }
        Ok(Some(code))
    }

    pub fn consume_join_code(&self) -> Result<(), Failure> {
        let _ = fs::remove_file(self.root.join("join-code"));
        Ok(())
    }

    pub fn set_accept_first(&self) -> Result<(), Failure> {
        self.private_write(&self.root.join("accept-first"), "")
    }

    pub fn accept_first(&self) -> bool {
        self.root.join("accept-first").is_file()
    }

    pub fn clear_accept_first(&self) {
        let _ = fs::remove_file(self.root.join("accept-first"));
    }
}

/// The HMAC key a join code yields: the SHA-256 of its normalized text.
pub fn join_key(code: &str) -> [u8; 32] {
    sha256(super::normalize_code(code).as_bytes())
}

/// A fresh join code: 20 random bytes as eight groups of four base32 characters.
pub fn new_join_code() -> Result<String, Failure> {
    let text = super::base32(&super::random(20)?);
    Ok(text
        .as_bytes()
        .chunks(4)
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect::<Vec<_>>()
        .join("-"))
}

// ---- controller --------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub name: String,
    pub fingerprint: String,
    pub public_key: String,
    pub address: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cluster {
    pub nodes: Vec<String>,
}

/// A group the operator wants: `flare orbit run --name NAME` takes its package and options
/// from here, and the fields are constraints for placement.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupSpec {
    pub name: String,
    /// A local path on the controller or `NODE:/absolute/path`.
    pub package: String,
    /// Serving options every rank loads with, as command-line words.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub serving: Vec<String>,
    /// Rank zero's HTTP bind and request options, as command-line words.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rank_zero: Vec<String>,
    /// Candidate nodes in rank order; empty lets placement choose from the pool.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<String>,
    /// NVML indices every member uses on its node; empty lets placement choose.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gpus: Vec<u32>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub roce_ring: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer_streams: Option<u8>,
    /// Serve over a chain in rank order; exclusive with `roce_ring`. A record without the field
    /// loads as not a chain.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub roce_chain: bool,
    /// Controller-local executable override; absent sends the controller's running binary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executable: Option<PathBuf>,
    /// Controller-local publisher profiles used to verify the package before each launch.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub publisher_keys: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust_store: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Orbit {
    /// The signed controller-set record every paired node holds.
    pub controllers: Option<Signed>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub node: Vec<Node>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub cluster: BTreeMap<String, Cluster>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group: Vec<GroupSpec>,
}

impl Node {
    pub fn key(&self) -> Result<PublicKey, Failure> {
        PublicKey::from_hex(&self.public_key)
    }
}

impl Orbit {
    pub fn default_path() -> PathBuf {
        home().join(".config/flare/orbit.toml")
    }

    pub fn default_key_path() -> PathBuf {
        home().join(".config/flare/orbit/controller.key")
    }

    pub fn load(path: &Path) -> Result<Self, Failure> {
        match fs::read_to_string(path) {
            Ok(text) => {
                toml::from_str(&text).map_err(|e| format!("invalid {}: {e}", path.display()).into())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(format!("cannot read {}: {error}", path.display()).into()),
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), Failure> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, toml::to_string(self)?)?;
        fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600))?;
        fs::rename(&temporary, path)?;
        Ok(())
    }

    pub fn controller_set(&self) -> Result<Option<ControllerSet>, Failure> {
        self.controllers
            .as_ref()
            .map(|record| record.verify(|_| true))
            .transpose()
    }

    pub fn node(&self, name: &str) -> Option<&Node> {
        self.node.iter().find(|n| n.name == name)
    }

    pub fn group(&self, name: &str) -> Option<&GroupSpec> {
        self.group.iter().find(|g| g.name == name)
    }

    pub fn save_group(&mut self, spec: GroupSpec) {
        if let Some(existing) = self.group.iter_mut().find(|group| group.name == spec.name) {
            *existing = spec;
        } else {
            self.group.push(spec);
        }
    }

    /// Record or update a node; a node is identified by its key.
    pub fn upsert(&mut self, node: Node) {
        if let Some(existing) = self
            .node
            .iter_mut()
            .find(|n| n.public_key == node.public_key)
        {
            *existing = node;
        } else {
            self.node.push(node);
        }
    }

    /// The nodes of a named cluster, in rank order; with no name, the orbit's only cluster.
    pub fn cluster(&self, name: Option<&str>) -> Result<(String, Vec<&Node>), Failure> {
        let (name, cluster) = match name {
            Some(name) => (
                name.to_owned(),
                self.cluster
                    .get(name)
                    .ok_or_else(|| format!("no cluster named {name} in the orbit file"))?,
            ),
            None if self.cluster.len() == 1 => {
                let (name, cluster) = self.cluster.iter().next().unwrap();
                (name.clone(), cluster)
            }
            None if self.cluster.is_empty() => {
                return Err(
                    "the orbit file defines no cluster; add [cluster.NAME] nodes = [...]".into(),
                );
            }
            None => {
                return Err("the orbit file defines several clusters; pass --cluster NAME".into());
            }
        };
        let nodes = cluster
            .nodes
            .iter()
            .map(|n| {
                self.node(n)
                    .ok_or_else(|| format!("cluster {name} names unknown node {n}").into())
            })
            .collect::<Result<Vec<_>, Failure>>()?;
        if nodes.is_empty() {
            return Err(format!("cluster {name} lists no nodes").into());
        }
        Ok((name, nodes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_codes_are_grouped_base32_and_expire() {
        let code = new_join_code().unwrap();
        assert_eq!(code.len(), 8 * 4 + 7, "{code}");
        assert!(code.split('-').all(|g| g.len() == 4));
        assert_eq!(
            join_key(&code),
            join_key(&code.to_lowercase().replace('-', " "))
        );
        let temp = tempfile::tempdir().unwrap();
        let dir = AgentDir::new(temp.path().join("agent"));
        dir.set_join_code(&code, 60).unwrap();
        assert_eq!(dir.join_code().unwrap().unwrap().key, join_key(&code));
        dir.set_join_code(&code, 0).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        assert!(dir.join_code().unwrap().is_none());
        assert_eq!(
            fs::metadata(&dir.root).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }

    #[test]
    fn orbit_files_round_trip_and_resolve_clusters() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("orbit.toml");
        let mut orbit = Orbit::load(&path).unwrap();
        assert!(orbit.node.is_empty());
        assert!(orbit.cluster(None).is_err());
        for name in ["a", "b"] {
            orbit.upsert(Node {
                name: name.into(),
                fingerprint: format!("f{name}"),
                public_key: name.repeat(64),
                address: format!("{name}:29900"),
            });
        }
        orbit.cluster.insert(
            "pair".into(),
            Cluster {
                nodes: vec!["b".into(), "a".into()],
            },
        );
        orbit.save(&path).unwrap();
        let loaded = Orbit::load(&path).unwrap();
        assert_eq!(loaded, orbit);
        let (name, nodes) = loaded.cluster(None).unwrap();
        assert_eq!(name, "pair");
        assert_eq!(
            nodes.iter().map(|n| n.name.as_str()).collect::<Vec<_>>(),
            ["b", "a"]
        );
        assert!(loaded.cluster(Some("other")).is_err());
    }

    #[test]
    fn persistent_group_definitions_survive_reload_and_update_by_name() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("orbit.toml");
        let mut orbit = Orbit::default();
        let mut group = GroupSpec {
            name: "service".into(),
            package: "node:/models/model.safetensors".into(),
            serving: vec!["--mtp-draft-tokens".into(), "5".into()],
            rank_zero: vec![
                "--host".into(),
                "0.0.0.0".into(),
                "--port".into(),
                "8001".into(),
            ],
            nodes: vec!["node".into()],
            gpus: vec![0],
            roce_ring: true,
            transfer_streams: Some(8),
            roce_chain: false,
            executable: Some("/bin/flare-pinned".into()),
            publisher_keys: vec!["/keys/publisher.json".into()],
            trust_store: Some("/keys/trust.toml".into()),
        };
        orbit.save_group(group.clone());
        orbit.save(&path).unwrap();
        let mut reloaded = Orbit::load(&path).unwrap();
        assert_eq!(reloaded.group("service"), Some(&group));
        group.serving = vec!["--mtp-draft-tokens".into(), "0".into()];
        reloaded.save_group(group.clone());
        reloaded.save(&path).unwrap();
        let reloaded = Orbit::load(&path).unwrap();
        assert_eq!(reloaded.group.len(), 1);
        assert_eq!(reloaded.group("service"), Some(&group));
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let manual: GroupSpec =
            toml::from_str("name = 'manual'\npackage = '/model.safetensors'").unwrap();
        assert!(manual.executable.is_none());
    }
}
