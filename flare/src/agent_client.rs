//! Paired agent identities and authenticated sessions shared by manual launch and Orbit.

use crate::orbit::{
    Failure, PROTOCOL_VERSION,
    agent::{Hello, exchange_hello},
    frame,
    grant::{ControllerKey, ControllerSet, Signed},
    hex, hmac_sha256, hostname,
    identity::{Key, PublicKey},
    session::Session,
    store::{Node, Orbit, join_key},
    unhex, with_port,
};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
/// The controlling machine's identity and orbit file.
pub struct Client {
    pub(crate) key: Key,
    pub(crate) orbit: Orbit,
    pub(crate) orbit_path: PathBuf,
    pub(crate) name: String,
}

impl Client {
    /// Load the controller key, creating it on first use, and the orbit file.
    pub fn open(orbit_path: Option<&Path>, key_path: Option<&Path>) -> Result<Self, Failure> {
        let key_path = key_path.map_or_else(Orbit::default_key_path, Path::to_path_buf);
        let orbit_path = orbit_path.map_or_else(Orbit::default_path, Path::to_path_buf);
        let key = Key::load_or_create(&key_path)
            .map_err(|e| format!("controller key {}: {e}", key_path.display()))?;
        let orbit = Orbit::load(&orbit_path)?;
        Ok(Self::new(key, orbit, orbit_path))
    }

    /// A controller with this key and orbit, saving the orbit to `orbit_path`.
    pub fn new(key: Key, orbit: Orbit, orbit_path: PathBuf) -> Self {
        Self {
            key,
            orbit,
            orbit_path,
            name: hostname(),
        }
    }

    pub(crate) fn save(&self) -> Result<(), Failure> {
        self.orbit.save(&self.orbit_path)
    }

    pub(crate) fn revision(&self) -> Result<u64, Failure> {
        Ok(self.orbit.controller_set()?.map_or(0, |set| set.revision))
    }

    // ---- sessions ------------------------------------------------------------------------------

    pub(crate) fn node_session(&self, node: &Node) -> Result<(Session, Hello), Failure> {
        let (mut session, mut hello) = self.handshake(&node.address)?;
        let expected = node.key()?;
        if session.remote() != expected {
            return Err(format!(
                "{} presents key {}, not the key this node paired with ({})",
                node.address,
                session.remote().fingerprint(),
                expected.fingerprint()
            )
            .into());
        }
        hello.controllers_revision = self.synchronize(&mut session, &hello)?;
        Ok((session, hello))
    }

    pub(crate) fn handshake(&self, address: &str) -> Result<(Session, Hello), Failure> {
        let mut session = Session::connect(address, &self.key)?;
        let mine = Hello {
            version: PROTOCOL_VERSION,
            kind: "controller".into(),
            name: self.name.clone(),
            public_key: self.key.public().to_hex(),
            controllers_revision: self.revision()?,
            free_bytes: 0,
            executable: None,
        };
        let hello = exchange_hello(&mut session, &mine)?;
        if hello.kind != "node" {
            return Err(format!("{address} is a flare {}, not an agent node", hello.kind).into());
        }
        Ok((session, hello))
    }

    /// Send the orbit file's controller-set record to a paired node holding an older revision,
    /// and return the revision the node holds afterwards. An unpaired node becomes paired only
    /// through PAIR, so it never receives the record here.
    pub(crate) fn synchronize(&self, session: &mut Session, hello: &Hello) -> Result<u64, Failure> {
        let ours = self.revision()?;
        let Some(record) = self.orbit.controllers.as_ref() else {
            return Ok(hello.controllers_revision);
        };
        if hello.controllers_revision == 0 || hello.controllers_revision >= ours {
            return Ok(hello.controllers_revision);
        }
        session.write_frame(frame::CONTROLLERS, &serde_json::to_vec(record)?)?;
        let reply = session
            .expect(frame::OK)
            .map_err(|e| format!("controller set revision {ours} refused: {e}"))?;
        let reply: Value = serde_json::from_slice(&reply)?;
        Ok(reply["revision"].as_u64().unwrap_or(ours))
    }

    // ---- pairing and the controller set --------------------------------------------------------

    /// Pair with the agents at `hosts` and record them in the orbit file. With `code`, both sides
    /// prove the join code inside the session; without it, each node must accept the first
    /// controller. Prints each node's name and fingerprint for comparison with what
    /// `flare agent install` printed on the host.
    pub fn pair(&mut self, hosts: &[String], code: Option<&str>) -> Result<(), Failure> {
        let record = self.pairing_record()?;
        let code_key = code.map(join_key);
        let mut failed = Vec::new();
        for host in hosts {
            match self.pair_one(&with_port(host), &record, code_key.as_ref()) {
                Ok(node) => println!(
                    "paired {}  fingerprint {}  address {}",
                    node.name, node.fingerprint, node.address
                ),
                Err(error) => {
                    eprintln!("{host}: {error}");
                    failed.push(host.as_str());
                }
            }
        }
        if !failed.is_empty() {
            return Err(format!("pairing failed for {}", failed.join(", ")).into());
        }
        Ok(())
    }

    pub(crate) fn pair_one(
        &mut self,
        address: &str,
        record: &Signed,
        code_key: Option<&[u8; 32]>,
    ) -> Result<Node, Failure> {
        let (mut session, hello) = self.handshake(address)?;
        let key = session.remote();
        if let Some(known) = self
            .orbit
            .node
            .iter()
            .find(|n| n.name == hello.name && n.public_key != key.to_hex())
        {
            return Err(format!(
                "the orbit file records another node named {} (fingerprint {}); remove it with \
                 `flare orbit forget {}` before pairing this host",
                known.name, known.fingerprint, known.name
            )
            .into());
        }
        let hash = *session.handshake_hash();
        let mut request = json!({"controllers": record});
        if let Some(code_key) = code_key {
            let proof = hmac_sha256(code_key, &[b"controller".as_slice(), &hash].concat());
            request["proof"] = hex(&proof).into();
        }
        let reply: Value = serde_json::from_slice(&session.request(frame::PAIR, &request)?)?;
        let public_key = PublicKey::from_hex(
            reply["public_key"]
                .as_str()
                .ok_or("PAIR reply lacks public_key")?,
        )?;
        if public_key != key {
            return Err(
                "the PAIR reply names a key other than the one the node authenticated".into(),
            );
        }
        if let Some(code_key) = code_key {
            let expected = hmac_sha256(code_key, &[b"node".as_slice(), &hash].concat());
            let proof = reply["proof"].as_str().map(unhex).transpose()?;
            if !proof.is_some_and(|proof| constant_time_equal(&proof, &expected)) {
                return Err(
                    "the node did not prove the join code, so it may not be the host where the \
                     code was installed; it was not recorded"
                        .into(),
                );
            }
        }
        let node = Node {
            name: reply["name"].as_str().unwrap_or(&hello.name).to_owned(),
            fingerprint: key.fingerprint(),
            public_key: key.to_hex(),
            address: address.to_owned(),
        };
        self.orbit.upsert(node.clone());
        self.save()?;
        Ok(node)
    }

    /// The orbit file's controller-set record, created at revision 1 with this controller alone
    /// when the orbit file has none.
    pub(crate) fn ensure_record(&mut self) -> Result<Signed, Failure> {
        if let Some(record) = &self.orbit.controllers {
            return Ok(record.clone());
        }
        let set = ControllerSet {
            revision: 1,
            keys: vec![ControllerKey {
                name: self.name.clone(),
                public_key: self.key.public().to_hex(),
            }],
        };
        let record = Signed::sign(&self.key, &set)?;
        self.orbit.controllers = Some(record.clone());
        self.save()?;
        Ok(record)
    }

    /// The record pairing hands a node. A node accepts a first record only from the controller
    /// that signed it, so a record another controller signed is signed again here with the same
    /// revision and keys.
    pub(crate) fn pairing_record(&mut self) -> Result<Signed, Failure> {
        let record = self.ensure_record()?;
        if record.signer()? == self.key.public() {
            return Ok(record);
        }
        let set: ControllerSet = record.verify(|_| true)?;
        if !set.contains(&self.key.public()) {
            return Err(self.not_a_member());
        }
        Signed::sign(&self.key, &set)
    }

    pub(crate) fn not_a_member(&self) -> Failure {
        format!(
            "this controller ({}) is not in the controller set of {}",
            self.key.public().fingerprint(),
            self.orbit_path.display()
        )
        .into()
    }

    /// Authorize another controller, given its public key as hex: write the next controller-set
    /// revision, signed by this controller, and send it to every node in the orbit file.
    pub fn controllers_add(&mut self, name: &str, public_key: &str) -> Result<(), Failure> {
        let key = PublicKey::from_hex(public_key.trim())
            .map_err(|e| format!("controller public key: {e}"))?;
        self.ensure_record()?;
        self.advance(|keys| {
            if keys.iter().any(|k| k.public_key == key.to_hex()) {
                return Err(
                    format!("{} is already an authorized controller", key.fingerprint()).into(),
                );
            }
            keys.push(ControllerKey {
                name: name.to_owned(),
                public_key: key.to_hex(),
            });
            Ok(())
        })
    }

    /// Remove the controller named by its fingerprint, public key, or name: write the next
    /// controller-set revision, signed by this controller, and send it to every node.
    pub fn controllers_remove(&mut self, controller: &str) -> Result<(), Failure> {
        self.advance(|keys| {
            let before = keys.len();
            keys.retain(|k| {
                !(k.name == controller
                    || k.public_key == controller
                    || PublicKey::from_hex(&k.public_key)
                        .is_ok_and(|p| p.fingerprint() == controller))
            });
            match before - keys.len() {
                1 => Ok(()),
                0 => Err(format!("no authorized controller matches {controller}").into()),
                _ => Err(format!(
                    "{controller} matches several controllers; name one by fingerprint"
                )
                .into()),
            }
        })
    }

    pub(crate) fn advance(
        &mut self,
        change: impl FnOnce(&mut Vec<ControllerKey>) -> Result<(), Failure>,
    ) -> Result<(), Failure> {
        let set = self
            .orbit
            .controller_set()?
            .ok_or("the orbit file has no controller set; pair a node first")?;
        if !set.contains(&self.key.public()) {
            return Err(self.not_a_member());
        }
        let mut keys = set.keys.clone();
        change(&mut keys)?;
        if keys.is_empty() {
            return Err("the controller set must keep at least one controller".into());
        }
        let next = ControllerSet {
            revision: set.revision + 1,
            keys,
        };
        self.orbit.controllers = Some(Signed::sign(&self.key, &next)?);
        self.save()?;
        println!("controller set revision {}:", next.revision);
        for key in &next.keys {
            let fingerprint = PublicKey::from_hex(&key.public_key)?.fingerprint();
            println!("  {fingerprint}  {}", key.name);
        }
        self.distribute();
        Ok(())
    }

    /// Bring every node in the orbit file to the orbit file's controller-set revision, reporting
    /// nodes that cannot be reached; those receive it when any controller next connects.
    pub(crate) fn distribute(&self) {
        let nodes = &self.orbit.node;
        let results = nodes.iter().map(|node| self.node_session(node));
        for (node, result) in nodes.iter().zip(results) {
            match result {
                Ok((_, hello)) => {
                    println!("  {}: revision {}", node.name, hello.controllers_revision)
                }
                Err(error) => eprintln!("  {}: not updated: {error}", node.name),
            }
        }
    }

    /// Remove a node from the orbit file. Clusters that list it fail to resolve until edited.
    pub fn forget(&mut self, name: &str) -> Result<(), Failure> {
        let before = self.orbit.node.len();
        self.orbit.node.retain(|n| n.name != name);
        if self.orbit.node.len() == before {
            return Err(format!("the orbit file has no node named {name}").into());
        }
        self.save()
    }

    /// Resolve explicitly listed paired peers without cluster placement or saved groups.
    pub fn peers(&self, hosts: &[String]) -> Result<Vec<Node>, Failure> {
        let mut result = Vec::new();
        for host in hosts {
            let address = with_port(host);
            let node = self
                .orbit
                .node
                .iter()
                .find(|n| n.name == *host || n.address == address)
                .ok_or_else(|| {
                    format!("agent {host} is not paired; use flare agent pair {host}")
                })?;
            if result
                .iter()
                .any(|n: &Node| n.public_key == node.public_key)
            {
                return Err(format!("agent {host} is listed more than once").into());
            }
            result.push(node.clone());
        }
        Ok(result)
    }
}
fn constant_time_equal(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |d, (x, y)| d | (x ^ y)) == 0
}
