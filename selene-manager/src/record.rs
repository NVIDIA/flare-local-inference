use crate::{Key, Result, invalid};
use blake2::{
    Blake2bVar,
    digest::{Update, VariableOutput},
};
use flare_selene::shared::record::{Header, RankSizes, checksum};

pub type NamespaceId = [u8; 32];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Descriptor {
    pub identity: [u8; 32],
    pub fingerprint: [u8; 16],
    pub page_tokens: u64,
    pub ranks: Vec<RankSizes>,
    pub pool_name: String,
}
impl Descriptor {
    pub fn matches(&self, header: &Header) -> bool {
        header.page_tokens == self.page_tokens && header.ranks == self.ranks
    }

    pub fn validate(&self) -> Result<()> {
        new_header(
            self,
            Key {
                consumed_tokens: self.page_tokens,
                digest: [0; 16],
            },
            None,
            [0; 16],
            1,
        )?;
        if self.pool_name.len() > 255 {
            return Err(invalid("pool name exceeds 255 bytes"));
        }
        Ok(())
    }
    pub fn namespace_id(&self) -> NamespaceId {
        namespace_id(&self.identity, &self.fingerprint, &self.pool_name)
    }
    pub fn name(&self) -> String {
        flare_selene::shared::record::directory_name(&self.identity, &self.fingerprint)
    }
    pub fn page_bytes(&self) -> u64 {
        self.ranks.iter().map(RankSizes::page_record_bytes).sum()
    }
    pub fn row_bytes(&self) -> u64 {
        self.ranks.iter().map(RankSizes::row_record_bytes).sum()
    }
}
pub fn new_header(
    descriptor: &Descriptor,
    key: Key,
    parent: Option<Key>,
    bucket: [u8; 16],
    tail_pages: u32,
) -> Result<Header> {
    Header::from_sizes(
        descriptor.page_tokens,
        descriptor.ranks.clone(),
        descriptor.identity,
        descriptor.fingerprint,
        key,
        parent,
        bucket,
        tail_pages,
        0,
    )
    .map_err(|e| invalid(e.to_string()))
}

pub fn namespace_id(identity: &[u8; 32], fingerprint: &[u8; 16], pool_name: &str) -> NamespaceId {
    let mut hasher = Blake2bVar::new(32).unwrap();
    hasher.update(b"selene-ns-v1");
    hasher.update(identity);
    hasher.update(fingerprint);
    hasher.update(&[pool_name.len() as u8]);
    hasher.update(pool_name.as_bytes());
    let mut id = [0; 32];
    hasher.finalize_variable(&mut id).unwrap();
    id
}
pub use flare_selene::shared::record::{file_name, parse_file_name};
pub fn verify_record(bytes: &[u8], header: &Header, row: bool) -> Result<()> {
    let length = header.length(row);
    if Header::decode(bytes).map_err(|e| invalid(e.to_string()))? != *header
        || bytes.len() as u64 != length
    {
        return Err(invalid("record header or length"));
    }
    for page in 0..header.tail_pages as usize {
        let start = header.page_segment as usize + page * header.page_record_bytes() as usize;
        let expected = u64::from_le_bytes(
            bytes[4096 + page * 8..4096 + page * 8 + 8]
                .try_into()
                .unwrap(),
        );
        if checksum(&bytes[start..start + header.page_record_bytes() as usize]) != expected {
            return Err(invalid("disk page checksum"));
        }
    }
    if row && checksum(&bytes[header.row_segment as usize..]) != header.row_checksum {
        return Err(invalid("disk row checksum"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flare_selene::shared::record::hex;
    fn key(tokens: u64) -> Key {
        Key {
            consumed_tokens: tokens,
            digest: (tokens as u128 * 7919).to_le_bytes(),
        }
    }
    fn descriptor() -> Descriptor {
        Descriptor {
            identity: [7; 32],
            fingerprint: [3; 16],
            page_tokens: 64,
            ranks: vec![RankSizes {
                planes: vec![4000],
                row_bytes: 1000,
            }],
            pool_name: "tenant".into(),
        }
    }
    #[test]
    fn persisted_header_matches_the_descriptor_layout() {
        let fixture = include_bytes!("../tests/fixtures/flare-header-v1.bin");
        let decoded = Header::decode(fixture).unwrap();
        let mut planned = new_header(&descriptor(), key(130), Some(key(65)), [4; 16], 1).unwrap();
        planned.created = 1_700_000_000;
        planned.row_checksum = 0x0123456789abcdef;
        assert_eq!(decoded, planned);
        assert_eq!(&planned.encode().unwrap(), fixture);
        for (offset, value) in [(129, 0u64), (173, 0), (181, u64::MAX)] {
            let mut invalid = *fixture;
            invalid[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            assert!(Header::decode(&invalid).is_err());
        }
    }
    #[test]
    fn namespace_id_vectors() {
        let id = namespace_id(&[0x63; 32], &[0x1b; 16], "test");
        assert_eq!(
            hex(&id),
            "83bbb4befe798c5d717456d55695acc1d166d45c214e54e253f249cd025d4040"
        );
        let id = namespace_id(&[0x63; 32], &[0x1b; 16], "");
        assert_eq!(
            hex(&id),
            "6fbf3a101213ca13f48c6ab6ae91e7e1e35fd409de023b30696e32b18c97ed74"
        );
    }
}
