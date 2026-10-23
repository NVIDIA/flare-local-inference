//! ELF transformations that preserve CUDA section and symbol indices.

use super::{ExportResult, invalid};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
struct Section {
    name: usize,
    kind: u32,
    flags: u64,
    offset: usize,
    size: usize,
    link: usize,
    stride: usize,
}
struct Elf<'a> {
    image: &'a [u8],
    offset: usize,
    names: usize,
    sections: Vec<Section>,
}
fn number(image: &[u8], offset: usize, size: usize) -> ExportResult<u64> {
    let bytes = image
        .get(
            offset
                ..offset
                    .checked_add(size)
                    .ok_or_else(|| invalid("invalid cubin extent"))?,
        )
        .ok_or_else(|| invalid("invalid cubin extent"))?;
    let mut value = [0; 8];
    value[..size].copy_from_slice(bytes);
    Ok(u64::from_le_bytes(value))
}
fn put(image: &mut [u8], offset: usize, value: u64, size: usize) {
    image[offset..offset + size].copy_from_slice(&value.to_le_bytes()[..size]);
}
fn overlap(a: usize, size: usize, b: usize, length: usize) -> bool {
    size > 0
        && length > 0
        && (a as u128) < b as u128 + length as u128
        && (b as u128) < a as u128 + size as u128
}
impl<'a> Elf<'a> {
    fn parse(image: &'a [u8]) -> ExportResult<Self> {
        if image.len() < 64 || &image[..6] != b"\x7fELF\x02\x01" {
            return Err(invalid("expected an ELF64 little-endian cubin"));
        }
        let offset = number(image, 40, 8)? as usize;
        let count = number(image, 60, 2)? as usize;
        let names = number(image, 62, 2)? as usize;
        if number(image, 58, 2)? != 64
            || names == 0
            || names >= count
            || offset
                .checked_add(count * 64)
                .is_none_or(|end| end > image.len())
        {
            return Err(invalid("invalid cubin ELF section table"));
        }
        let mut sections = Vec::with_capacity(count);
        for i in 0..count {
            let p = offset + i * 64;
            sections.push(Section {
                name: number(image, p, 4)? as usize,
                kind: number(image, p + 4, 4)? as u32,
                flags: number(image, p + 8, 8)?,
                offset: number(image, p + 24, 8)? as usize,
                size: number(image, p + 32, 8)? as usize,
                link: number(image, p + 40, 4)? as usize,
                stride: number(image, p + 56, 8)? as usize,
            });
        }
        Ok(Self {
            image,
            offset,
            names,
            sections,
        })
    }
    fn contents(&self, index: usize) -> ExportResult<&'a [u8]> {
        let s = self
            .sections
            .get(index)
            .ok_or_else(|| invalid("invalid cubin section reference"))?;
        if s.kind == 8 {
            return Err(invalid("invalid cubin section extent"));
        }
        self.image
            .get(
                s.offset
                    ..s.offset
                        .checked_add(s.size)
                        .ok_or_else(|| invalid("invalid cubin section extent"))?,
            )
            .ok_or_else(|| invalid("invalid cubin section extent"))
    }
    fn string(&self, table: usize, index: usize) -> ExportResult<&'a [u8]> {
        if self.sections.get(table).is_none_or(|s| s.kind != 3) {
            return Err(invalid("invalid cubin string reference"));
        }
        let data = self
            .contents(table)?
            .get(index..)
            .ok_or_else(|| invalid("invalid cubin string reference"))?;
        let end = data
            .iter()
            .position(|b| *b == 0)
            .ok_or_else(|| invalid("unterminated cubin string"))?;
        Ok(&data[..end])
    }
    fn symbols(
        &self,
        alternate: bool,
    ) -> ExportResult<Vec<(usize, usize, &'a [u8], u8, usize, u64, usize)>> {
        let mut result = Vec::new();
        for (index, s) in self.sections.iter().enumerate() {
            if s.kind != 2 && !(alternate && s.kind == 0x70000085) {
                continue;
            }
            if s.link == 0 || s.link >= self.sections.len() || s.stride != 24 || s.size % 24 != 0 {
                return Err(invalid("invalid cubin symbol table"));
            }
            self.contents(index)
                .map_err(|_| invalid("invalid cubin symbol table extent"))?;
            for p in (s.offset..s.offset + s.size).step_by(24) {
                result.push((
                    p,
                    s.link,
                    self.string(s.link, number(self.image, p, 4)? as usize)?,
                    self.image[p + 4] & 15,
                    number(self.image, p + 6, 2)? as usize,
                    number(self.image, p + 8, 8)?,
                    number(self.image, p + 16, 8)? as usize,
                ));
            }
        }
        Ok(result)
    }
}

pub(crate) fn strip_global_source(symbol: &str, data: &[u8]) -> ExportResult<Vec<u8>> {
    if !symbol.starts_with("assertFile_") {
        return Ok(data.to_vec());
    }
    if data.last() != Some(&0) {
        return Err(invalid(
            "CUDA assertion filename must be a terminated string",
        ));
    }
    let mut result = vec![0; data.len()];
    let length = 6.min(data.len() - 1);
    result[..length].copy_from_slice(&b"kernel"[..length]);
    Ok(result)
}

pub(crate) fn strip_debug(image: &[u8]) -> ExportResult<Vec<u8>> {
    let elf = Elf::parse(image)?;
    let mut debug = BTreeSet::new();
    for (i, s) in elf.sections.iter().enumerate() {
        let name = elf.string(elf.names, s.name)?;
        if [b".debug".as_slice(), b".zdebug", b".nv_debug"]
            .iter()
            .any(|needle| name.windows(needle.len()).any(|w| w == *needle))
            || name == b".comment"
        {
            debug.insert(i);
        }
    }
    let mut result = image.to_vec();
    for &i in &debug {
        let s = &elf.sections[i];
        if s.flags & 2 != 0
            || s.offset
                .checked_add(s.size)
                .is_none_or(|end| end > image.len())
        {
            return Err(invalid("invalid or device-allocated cubin debug section"));
        }
        for (j, t) in elf.sections.iter().enumerate() {
            if !debug.contains(&j) && t.kind != 8 && overlap(s.offset, s.size, t.offset, t.size) {
                return Err(invalid("cubin debug section overlaps executable metadata"));
            }
        }
        result[s.offset..s.offset + s.size].fill(0);
        put(&mut result, elf.offset + i * 64 + 32, 0, 8);
    }
    for (_, _, name, kind, index, value, size) in elf.symbols(true)? {
        if kind != 1
            || !name.starts_with(b"assertFile_")
            || index == 0
            || index >= elf.sections.len()
        {
            continue;
        }
        let s = &elf.sections[index];
        if s.kind == 8
            || value
                .checked_add(size as u64)
                .is_none_or(|end| end > s.size as u64)
        {
            return Err(invalid("invalid CUDA assertion filename symbol"));
        }
        elf.contents(index)?;
        let start = s.offset + value as usize;
        let clean = strip_global_source(std::str::from_utf8(name)?, &image[start..start + size])?;
        result[start..start + size].copy_from_slice(&clean);
    }
    Ok(result)
}

pub(crate) fn global_symbols(image: &[u8]) -> ExportResult<Vec<String>> {
    let elf = Elf::parse(image)?;
    let mut result = BTreeSet::new();
    for (_, _, name, kind, index, _, size) in elf.symbols(false)? {
        if kind != 1 || size == 0 || index == 0 || index >= elf.sections.len() {
            continue;
        }
        let section_name = elf.string(elf.names, elf.sections[index].name)?;
        if section_name.starts_with(b".nv.global") || section_name.starts_with(b".nv.constant") {
            result.insert(std::str::from_utf8(name)?.to_owned());
        }
    }
    Ok(result.into_iter().collect())
}

pub(crate) fn redact_symbols(image: &[u8]) -> ExportResult<(Vec<u8>, BTreeMap<String, String>)> {
    let elf = Elf::parse(image)?;
    let symbols = elf.symbols(true)?;
    let mut existing: BTreeSet<Vec<u8>> = symbols.iter().map(|s| s.2.to_vec()).collect();
    let mut aliases: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    for (_, _, name, kind, index, _, _) in &symbols {
        if !matches!(kind, 1 | 2)
            || *index == 0
            || *index >= elf.sections.len()
            || name.is_empty()
            || [b".nv.".as_slice(), b"__nv_", b"__cuda"]
                .iter()
                .any(|prefix| name.starts_with(prefix))
            || aliases.contains_key(*name)
        {
            continue;
        }
        let mut alias = format!("_f{}", aliases.len()).into_bytes();
        while existing.contains(&alias) {
            alias.insert(0, b'_');
        }
        existing.insert(alias.clone());
        aliases.insert(name.to_vec(), alias);
    }
    let renamed = |name: &[u8]| -> Vec<u8> {
        if let Some(alias) = aliases.get(name) {
            return alias.clone();
        }
        for (i, b) in name.iter().enumerate() {
            if *b == b'.' {
                if let Some(alias) = aliases.get(&name[i + 1..]) {
                    let mut value = name[..i + 1].to_vec();
                    value.extend(alias);
                    return value;
                }
            }
        }
        name.to_vec()
    };
    let mut tables: BTreeMap<usize, (Vec<u8>, BTreeMap<Vec<u8>, usize>)> = BTreeMap::new();
    fn intern(
        tables: &mut BTreeMap<usize, (Vec<u8>, BTreeMap<Vec<u8>, usize>)>,
        table: usize,
        value: Vec<u8>,
    ) -> usize {
        let (data, entries) = tables
            .entry(table)
            .or_insert_with(|| (vec![0], BTreeMap::from([(vec![], 0)])));
        *entries.entry(value.clone()).or_insert_with(|| {
            let offset = data.len();
            data.extend(value);
            data.push(0);
            offset
        })
    }
    let mut table_order = vec![elf.names];
    for symbol in &symbols {
        if !table_order.contains(&symbol.1) {
            table_order.push(symbol.1);
        }
    }
    let mut result = image.to_vec();
    for (i, s) in elf.sections.iter().enumerate() {
        let index = intern(
            &mut tables,
            elf.names,
            renamed(elf.string(elf.names, s.name)?),
        );
        put(&mut result, elf.offset + i * 64, index as u64, 4);
    }
    for (p, table, name, kind, _, _, _) in symbols {
        let index = intern(
            &mut tables,
            table,
            if kind == 4 { vec![] } else { renamed(name) },
        );
        put(&mut result, p, index as u64, 4);
    }
    let program = number(image, 32, 8)? as usize;
    let stride = number(image, 54, 2)? as usize;
    let count = number(image, 56, 2)? as usize;
    if count > 0
        && (stride != 56
            || program
                .checked_add(count * stride)
                .is_none_or(|end| end > image.len()))
    {
        return Err(invalid("invalid cubin program table"));
    }
    for index in table_order {
        let (data, _) = tables.remove(&index).unwrap();
        let s = &elf.sections[index];
        if s.flags & 2 != 0 {
            return Err(invalid(
                "cannot redact a device-allocated cubin string table",
            ));
        }
        if overlap(s.offset, s.size, elf.offset, elf.sections.len() * 64) {
            return Err(invalid("cubin string table overlaps section headers"));
        }
        if s.offset < 64 || overlap(s.offset, s.size, program, count * stride) {
            return Err(invalid("cubin string table overlaps ELF headers"));
        }
        for (i, t) in elf.sections.iter().enumerate() {
            if i != index && t.kind != 8 && overlap(s.offset, s.size, t.offset, t.size) {
                return Err(invalid("cubin string table overlaps another section"));
            }
        }
        for i in 0..count {
            let p = program + i * stride;
            if number(image, p, 4)? == 1
                && overlap(
                    s.offset,
                    s.size,
                    number(image, p + 8, 8)? as usize,
                    number(image, p + 32, 8)? as usize,
                )
            {
                return Err(invalid("cannot redact a loadable cubin string table"));
            }
        }
        result[s.offset..s.offset + s.size].fill(0);
        let end = result.len();
        put(&mut result, elf.offset + index * 64 + 24, end as u64, 8);
        put(
            &mut result,
            elf.offset + index * 64 + 32,
            data.len() as u64,
            8,
        );
        result.extend(data);
    }
    let aliases = aliases
        .into_iter()
        .map(|(key, value)| Ok((String::from_utf8(key)?, String::from_utf8(value)?)))
        .collect::<ExportResult<_>>()?;
    Ok((result, aliases))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cubin() -> Vec<u8> {
        let names = b"\0.strtab\0.symtab\0.text.descriptive_kernel\0.shstrtab\0.nv.global\0.debug_info\0.nv.info.descriptive_kernel\0";
        let strings = b"\0/private/source/descriptive_kernel\0weight_layout\0__assertfail\0.nv.reservedSmem.offset0\0";
        let find =
            |data: &[u8], name: &[u8]| data.windows(name.len()).position(|w| w == name).unwrap();
        let mut symbols = vec![0; 24];
        for (name, kind, index, size) in [
            (b"/private/source".as_slice(), 4u8, 0xfff1u64, 0),
            (b"descriptive_kernel", 0x12, 3, 16),
            (b"weight_layout", 0x11, 5, 8),
            (b"__assertfail", 0x12, 0, 0),
            (b".nv.reservedSmem.offset0", 0x11, 5, 8),
        ] {
            let mut symbol = vec![0; 24];
            put(&mut symbol, 0, find(strings, name) as u64, 4);
            symbol[4] = kind;
            put(&mut symbol, 6, index, 2);
            put(&mut symbol, 16, size, 8);
            symbols.extend(symbol);
        }
        let entries = [
            (b"".as_slice(), 0, 0, vec![], 0, 0),
            (b".strtab", 3, 0, strings.to_vec(), 0, 0),
            (b".symtab", 2, 0, symbols, 1, 24),
            (b".text.descriptive_kernel", 1, 6, (0..16).collect(), 0, 0),
            (b".shstrtab", 3, 0, names.to_vec(), 0, 0),
            (b".nv.global", 1, 3, b"constant".to_vec(), 0, 0),
            (b".debug_info", 1, 0, b"/private/kernel.py\0".to_vec(), 0, 0),
            (
                b".nv.info.descriptive_kernel",
                0x70000000,
                0,
                b"cuda metadata".to_vec(),
                2,
                0,
            ),
        ];
        let mut image = vec![0; 64 + entries.len() * 64];
        image[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
        put(&mut image, 40, 64, 8);
        put(&mut image, 54, 56, 2);
        put(&mut image, 58, 64, 2);
        put(&mut image, 60, entries.len() as u64, 2);
        put(&mut image, 62, 4, 2);
        for (i, (name, kind, flags, data, link, stride)) in entries.iter().enumerate() {
            let p = 64 + i * 64;
            put(
                &mut image,
                p,
                if name.is_empty() {
                    0
                } else {
                    find(names, name) as u64
                },
                4,
            );
            put(&mut image, p + 4, *kind, 4);
            put(&mut image, p + 8, *flags, 8);
            let offset = image.len();
            put(&mut image, p + 24, offset as u64, 8);
            put(&mut image, p + 32, data.len() as u64, 8);
            put(&mut image, p + 40, *link, 4);
            put(&mut image, p + 56, *stride, 8);
            image.extend(data);
        }
        image
    }
    #[test]
    fn redaction_preserves_executable_layout_and_removes_source_names() {
        let original = cubin();
        let stripped = strip_debug(&original).unwrap();
        let (clean, aliases) = redact_symbols(&stripped).unwrap();
        assert_eq!(
            aliases,
            BTreeMap::from([
                ("descriptive_kernel".into(), "_f0".into()),
                ("weight_layout".into(), "_f1".into())
            ])
        );
        for secret in [
            b"descriptive_kernel".as_slice(),
            b"weight_layout",
            b"/private/",
        ] {
            assert!(!clean.windows(secret.len()).any(|w| w == secret));
        }
        assert_eq!(
            global_symbols(&clean).unwrap(),
            vec![".nv.reservedSmem.offset0", "_f1"]
        );
        let before = Elf::parse(&original).unwrap();
        let after = Elf::parse(&clean).unwrap();
        for i in [3, 5, 7] {
            assert_eq!(before.contents(i).unwrap(), after.contents(i).unwrap());
            assert_eq!(before.sections[i].offset, after.sections[i].offset);
        }
        assert_eq!(after.sections[6].size, 0);
        assert!(
            clean[before.sections[6].offset..before.sections[6].offset + before.sections[6].size]
                .iter()
                .all(|b| *b == 0)
        );
        assert_eq!(after.sections[4].offset, original.len());
        assert!(after.sections[1].offset > after.sections[4].offset);
    }
    #[test]
    fn redaction_rejects_tables_that_overlap_executable_data() {
        let original = cubin();
        let elf = Elf::parse(&original).unwrap();
        let mut image = original.clone();
        put(
            &mut image,
            64 + 64 + 32,
            (elf.sections[1].size + 1) as u64,
            8,
        );
        assert!(
            redact_symbols(&image)
                .unwrap_err()
                .to_string()
                .contains("overlaps another section")
        );
    }
    #[test]
    fn debug_stripping_redacts_embedded_assertion_filenames() {
        let mut image = cubin();
        let symbol = image
            .windows(13)
            .position(|w| w == b"weight_layout")
            .unwrap();
        image[symbol..symbol + 13].copy_from_slice(b"assertFile_1\0");
        assert!(
            strip_debug(&image)
                .unwrap_err()
                .to_string()
                .contains("terminated string")
        );
        let start = Elf::parse(&image).unwrap().sections[5].offset;
        image[start..start + 8].copy_from_slice(b"/tmp/a\0\0");
        let clean = strip_debug(&image).unwrap();
        assert_eq!(&clean[start..start + 8], b"kernel\0\0");
    }

    #[test]
    fn assertion_source_redaction_preserves_size_and_termination() {
        for size in 1..20 {
            let mut source = vec![b'x'; size];
            source[size - 1] = 0;
            let clean = strip_global_source("assertFile_1", &source).unwrap();
            assert_eq!(clean.len(), size);
            assert_eq!(*clean.last().unwrap(), 0);
            assert!(clean.starts_with(&b"kernel"[..6.min(size - 1)]));
        }
        assert!(strip_global_source("assertFile_1", b"unterminated").is_err());
        assert_eq!(
            strip_global_source("other", b"/private/file").unwrap(),
            b"/private/file"
        );
    }
}
