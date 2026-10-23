//! Signed records: launch grants and the controller set.
//!
//! A [`Signed`] value carries the JSON encoding of its body exactly as the signer serialized it,
//! the signer's public key, and the signature over those bytes. Verification checks the signature
//! before parsing the body, so a record is trusted only in the form that was signed.

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::identity::{Key, PublicKey};
use super::{Failure, hex, unhex};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Signed {
    /// The body as serialized by the signer.
    pub body: String,
    /// The signer's public key, hex.
    pub signer: String,
    /// Ed25519 signature over `body`, hex.
    pub signature: String,
}

impl Signed {
    pub fn sign<T: Serialize>(key: &Key, value: &T) -> Result<Self, Failure> {
        let body = serde_json::to_string(value)?;
        Ok(Self {
            signature: hex(&key.sign(body.as_bytes())),
            signer: key.public().to_hex(),
            body,
        })
    }

    pub fn signer(&self) -> Result<PublicKey, Failure> {
        PublicKey::from_hex(&self.signer)
    }

    /// The body, when `authorized` admits the signer and the signature verifies.
    pub fn verify<T: DeserializeOwned>(
        &self,
        authorized: impl Fn(&PublicKey) -> bool,
    ) -> Result<T, Failure> {
        let signer = self.signer()?;
        if !authorized(&signer) {
            return Err(format!("signer {} is not authorized", signer.fingerprint()).into());
        }
        signer.verify(self.body.as_bytes(), &unhex(&self.signature)?)?;
        Ok(serde_json::from_str(&self.body)?)
    }
}

/// One rank's host in a launch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Member {
    pub rank: usize,
    pub name: String,
    pub public_key: String,
    pub address: String,
    /// UUIDs of the GPUs the rank uses on its node; empty leaves the choice to the rank.
    #[serde(default)]
    pub gpus: Vec<String>,
}

/// Where a node-held package lives: the node's public key and the primary file's absolute path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    pub public_key: String,
    pub path: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Package {
    /// The export identity every rank's file records.
    pub export: String,
    /// The primary file's length in bytes; cache entries carry it in their names.
    pub length: u64,
    /// The primary file's basename, then its companions.
    pub files: Vec<String>,
    /// Set when a node holds the package; absent when the controller sends it.
    pub source: Option<Source>,
}

/// The controller's signed statement of one launch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grant {
    pub group: String,
    /// The operator's name for the group, stable across launches.
    #[serde(default)]
    pub name: String,
    pub package: Package,
    pub members: Vec<Member>,
    /// Rank zero's rendezvous address as peers reach it, `IP:PORT`.
    pub rendezvous: String,
    /// Serving options every rank loads with.
    pub serving: Vec<String>,
    /// Options for rank zero alone: the HTTP bind and request handling.
    pub rank_zero: Vec<String>,
    pub roce_ring: bool,
    /// Fabric TCP endpoints for direct peer transfers or neighbor relays.
    #[serde(default)]
    pub transfer_fabric: Option<crate::fabric::Topology>,
    /// True for a single-host tensor-parallel package: one member runs every rank in one process
    /// over its GPUs, with no rendezvous.
    #[serde(default)]
    pub single_host: bool,
    /// Serve over a chain in rank order; a grant without the field is not a chain.
    #[serde(default)]
    pub roce_chain: bool,
    pub issued: u64,
    pub expires: u64,
}

impl Grant {
    pub fn member(&self, key: &PublicKey) -> Option<&Member> {
        let hex = key.to_hex();
        self.members.iter().find(|m| m.public_key == hex)
    }

    pub fn rank(&self, rank: usize) -> Option<&Member> {
        self.members.iter().find(|m| m.rank == rank)
    }

    /// Whether the grant still authorizes a launch at `now`.
    pub fn current(&self, now: u64) -> bool {
        now <= self.expires
    }

    /// The cache directory holding this package's files for `rank`.
    pub fn cache_directory(&self, rank: usize) -> String {
        format!("{}-{}-rank{rank}", self.package.export, self.package.length)
    }

    /// The cache directory holding the complete package.
    pub fn complete_directory(&self) -> String {
        format!("{}-{}", self.package.export, self.package.length)
    }

    pub fn primary(&self) -> &str {
        &self.package.files[0]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControllerKey {
    pub name: String,
    pub public_key: String,
}

/// The set of controllers a node obeys, at a revision. Revision 1 is created by pairing; each
/// later revision is signed by a key present in the revision before it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControllerSet {
    pub revision: u64,
    pub keys: Vec<ControllerKey>,
}

impl ControllerSet {
    pub fn contains(&self, key: &PublicKey) -> bool {
        let hex = key.to_hex();
        self.keys.iter().any(|k| k.public_key == hex)
    }

    pub fn public_keys(&self) -> Result<Vec<PublicKey>, Failure> {
        self.keys
            .iter()
            .map(|k| PublicKey::from_hex(&k.public_key))
            .collect()
    }
}

/// Verify a controller-set record against the set it replaces: a first record needs a signer in
/// its own list; a later one needs a higher revision and a signer from the stored set.
pub fn verify_controller_set(
    record: &Signed,
    stored: Option<&ControllerSet>,
) -> Result<ControllerSet, Failure> {
    let set: ControllerSet = match stored {
        None => record.verify(|_| true)?,
        Some(stored) => record.verify(|signer| stored.contains(signer))?,
    };
    if set.keys.is_empty() {
        return Err("a controller set must list at least one key".into());
    }
    set.public_keys()?;
    match stored {
        None => {
            let signer = record.signer()?;
            if !set.contains(&signer) {
                return Err("a controller set's first record must list its signer".into());
            }
        }
        Some(stored) if set.revision <= stored.revision => {
            return Err(format!(
                "controller set revision {} does not exceed the stored revision {}",
                set.revision, stored.revision
            )
            .into());
        }
        Some(_) => {}
    }
    Ok(set)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grant() -> Grant {
        Grant {
            group: "g".into(),
            name: "g".into(),
            package: Package {
                export: "export".into(),
                length: 10,
                files: vec!["model.safetensors".into()],
                source: None,
            },
            members: vec![Member {
                rank: 0,
                name: "n".into(),
                public_key: Key::from_seed(&[3; 32]).unwrap().public().to_hex(),
                address: "n:29900".into(),
                gpus: vec![],
            }],
            rendezvous: "10.0.0.1:29800".into(),
            serving: vec![],
            rank_zero: vec![],
            roce_ring: false,
            transfer_fabric: None,
            single_host: false,
            roce_chain: false,
            issued: 100,
            expires: 700,
        }
    }

    #[test]
    fn grants_verify_only_from_authorized_signers_and_unmodified() {
        let controller = Key::from_seed(&[1; 32]).unwrap();
        let other = Key::from_seed(&[2; 32]).unwrap();
        let signed = Signed::sign(&controller, &grant()).unwrap();
        let verified: Grant = signed.verify(|k| *k == controller.public()).unwrap();
        assert_eq!(verified, grant());
        assert!(signed.verify::<Grant>(|k| *k == other.public()).is_err());
        let mut forged = signed.clone();
        forged.body = forged.body.replace("\"expires\":700", "\"expires\":900");
        assert!(forged.verify::<Grant>(|_| true).is_err());
        let mut reissued = signed.clone();
        reissued.signer = other.public().to_hex();
        assert!(reissued.verify::<Grant>(|_| true).is_err());
        assert!(verified.current(700) && !verified.current(701));
        assert_eq!(verified.cache_directory(2), "export-10-rank2");
        assert!(
            verified
                .member(&Key::from_seed(&[3; 32]).unwrap().public())
                .is_some()
        );
    }

    #[test]
    fn controller_sets_advance_only_through_their_members() {
        let first = Key::from_seed(&[1; 32]).unwrap();
        let second = Key::from_seed(&[2; 32]).unwrap();
        let stranger = Key::from_seed(&[9; 32]).unwrap();
        let key = |k: &Key, name: &str| ControllerKey {
            name: name.into(),
            public_key: k.public().to_hex(),
        };
        let revision1 = ControllerSet {
            revision: 1,
            keys: vec![key(&first, "first")],
        };
        // The first record must list its own signer.
        let bad = Signed::sign(&stranger, &revision1).unwrap();
        assert!(verify_controller_set(&bad, None).is_err());
        let record1 = Signed::sign(&first, &revision1).unwrap();
        let stored = verify_controller_set(&record1, None).unwrap();
        let revision2 = ControllerSet {
            revision: 2,
            keys: vec![key(&first, "first"), key(&second, "second")],
        };
        // Only a member of the stored set advances it, and only to a higher revision.
        assert!(
            verify_controller_set(&Signed::sign(&second, &revision2).unwrap(), Some(&stored))
                .is_err()
        );
        let stored =
            verify_controller_set(&Signed::sign(&first, &revision2).unwrap(), Some(&stored))
                .unwrap();
        let revision3 = ControllerSet {
            revision: 3,
            keys: vec![key(&second, "second")],
        };
        let stored =
            verify_controller_set(&Signed::sign(&second, &revision3).unwrap(), Some(&stored))
                .unwrap();
        // A removed controller cannot produce an accepted record, whatever list it carries.
        let revision4 = ControllerSet {
            revision: 4,
            keys: vec![key(&first, "first")],
        };
        assert!(
            verify_controller_set(&Signed::sign(&first, &revision4).unwrap(), Some(&stored))
                .is_err()
        );
        assert!(
            verify_controller_set(&Signed::sign(&second, &revision3).unwrap(), Some(&stored))
                .is_err()
        );
    }
}
