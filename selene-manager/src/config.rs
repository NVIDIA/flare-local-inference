//! Manager configuration: command line and validation.
//!
//! This module is self-contained so it can be unit-tested without starting a manager.

use std::{io, net::SocketAddr, path::PathBuf};

pub const DEFAULT_NAME: &str = "default";
pub const DEFAULT_MEMORY: u64 = 1 << 30;
pub const DEFAULT_STAGING: u64 = 4 << 30;
pub const MIN_MEMORY: u64 = 4096;
pub const MIN_STAGING: u64 = 8 << 20;
pub const MIN_NVME: u64 = 8192;
pub const MAX_NAME: usize = 64;
pub const MAX_SOCKET: usize = 107;

#[derive(Clone, Debug)]
pub struct Config {
    pub name: String,
    pub socket: PathBuf,
    pub socket_explicit: bool,
    pub root: Option<PathBuf>,
    pub memory: u64,
    pub nvme: u64,
    pub staging: u64,
    /// Serve the pool protocol to instances on other hosts at this address.
    pub listen: Option<SocketAddr>,
    /// Where those instances connect to copy bytes. Defaults to the host of `listen` on
    /// the next port; set it when that port is taken or when it is not the one clients reach.
    pub listen_data: Option<SocketAddr>,
    /// TLS for `listen`: the manager's certificate chain and key. Clients verify the
    /// certificate against their configured CA; there are no client certificates.
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    /// Offer the RDMA data transport. Experimental; needs the `rdma` build feature and
    /// cannot be combined with TLS.
    pub rdma: bool,
    pub metrics: Option<SocketAddr>,
    #[cfg(any(test, feature = "test-support"))]
    pub faults: crate::disk::Faults,
    #[cfg(any(test, feature = "test-support"))]
    pub limits: flare_selene::shared::Limits,
}

impl Default for Config {
    fn default() -> Self {
        Self::defaults()
    }
}

impl Config {
    pub fn new(root: PathBuf, socket: PathBuf, memory: usize, nvme: u64) -> Self {
        Self {
            root: Some(root),
            socket,
            socket_explicit: true,
            memory: memory as u64,
            nvme,
            staging: crate::store::EMBEDDED_STAGING as u64,
            ..Self::defaults()
        }
    }
    pub fn dram_only(socket: PathBuf, memory: usize) -> Self {
        Self {
            socket,
            socket_explicit: true,
            memory: memory as u64,
            staging: crate::store::EMBEDDED_STAGING as u64,
            ..Self::defaults()
        }
    }

    pub fn defaults() -> Self {
        Self {
            name: DEFAULT_NAME.into(),
            socket: PathBuf::from(format!("/run/selene/{DEFAULT_NAME}.sock")),
            socket_explicit: false,
            root: None,
            memory: DEFAULT_MEMORY,
            nvme: 0,
            staging: DEFAULT_STAGING,
            listen: None,
            listen_data: None,
            tls_cert: None,
            tls_key: None,
            rdma: false,
            metrics: None,
            #[cfg(any(test, feature = "test-support"))]
            faults: Default::default(),
            #[cfg(any(test, feature = "test-support"))]
            limits: Default::default(),
        }
    }

    /// Default socket path for a manager name.
    pub fn default_socket(name: &str) -> PathBuf {
        PathBuf::from(format!("/run/selene/{name}.sock"))
    }

    /// Stable region namespace for this pool and Unix listener, including across restarts.
    pub fn region_name(&self) -> io::Result<String> {
        use blake2::{Blake2b, Digest, digest::consts::U16};
        use std::os::unix::ffi::OsStrExt;

        let socket = std::path::absolute(&self.socket)?;
        let digest = Blake2b::<U16>::digest(socket.as_os_str().as_bytes());
        Ok(format!("{}-{digest:x}", self.name))
    }

    pub fn shm_directory(&self) -> io::Result<PathBuf> {
        Ok(PathBuf::from(crate::local::SHM_ROOT).join(self.region_name()?))
    }

    /// The data listener's address: `listen_data`, else the next port of `listen`.
    pub fn data_address(&self) -> io::Result<Option<SocketAddr>> {
        let Some(listen) = self.listen else {
            return Ok(None);
        };
        if let Some(address) = self.listen_data {
            return Ok(Some(address));
        }
        // Port 0 asks for any free port, and so does its data listener.
        let port = match listen.port() {
            0 => 0,
            port => port.checked_add(1).ok_or_else(|| {
                invalid("--listen-data is required when --listen's port is 65535")
            })?,
        };
        Ok(Some(SocketAddr::new(listen.ip(), port)))
    }

    /// Whether this build and configuration offer RDMA.
    pub fn rdma_enabled(&self) -> bool {
        cfg!(feature = "rdma") && self.rdma
    }

    /// One line per opt-in that widens the manager's exposure, printed once at startup.
    pub fn startup_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if let Some(address) = self.listen {
            let mode = if self.tls_cert.is_some() {
                "TLS"
            } else {
                "plaintext"
            };
            let data = self
                .data_address()
                .ok()
                .flatten()
                .map(|data| format!(", data on {data}"))
                .unwrap_or_default();
            warnings.push(format!(
                "selene: warning: network listening enabled on {address} ({mode}{data})"
            ));
        }
        if self.rdma_enabled() {
            warnings.push("selene: warning: RDMA enabled (experimental)".to_string());
        }
        warnings
    }

    /// Validate every configuration rule.
    pub fn validate(&self) -> io::Result<()> {
        if !valid_name(&self.name) {
            return Err(invalid(
                "--name must be 1-64 bytes of letters, digits, '.', '_' or '-'",
            ));
        }
        if self.socket.as_os_str().len() > MAX_SOCKET {
            return Err(invalid("--socket path must be at most 107 bytes"));
        }
        if self.memory < MIN_MEMORY || !self.memory.is_multiple_of(4096) {
            return Err(invalid(
                "--memory must be at least 4096 bytes and a multiple of 4096",
            ));
        }
        if self.staging < MIN_STAGING || !self.staging.is_multiple_of(4096) {
            return Err(invalid(
                "--staging must be at least 8 MiB and a multiple of 4096",
            ));
        }
        match &self.root {
            Some(_) if self.nvme < MIN_NVME => {
                return Err(invalid("--nvme must be at least 8192 when --root is set"));
            }
            None if self.nvme != 0 => {
                return Err(invalid("--nvme requires --root"));
            }
            _ => {}
        }
        if self.listen_data.is_some() && self.listen.is_none() {
            return Err(invalid("--listen-data applies to --listen"));
        }
        if self.tls_cert.is_some() != self.tls_key.is_some() {
            return Err(invalid("--tls-cert and --tls-key go together"));
        }
        if self.tls_cert.is_some() && self.listen.is_none() {
            return Err(invalid("--tls-cert and --tls-key apply to --listen"));
        }
        if self.rdma && !cfg!(feature = "rdma") {
            return Err(invalid("--rdma requires a build with the `rdma` feature"));
        }
        if self.rdma && self.tls_cert.is_some() {
            return Err(invalid(
                "RDMA transfers are not encrypted, so --rdma cannot be combined with TLS; \
                 drop --rdma to keep TLS, or drop the TLS options to use RDMA",
            ));
        }
        if self.rdma && self.listen.is_none() {
            return Err(invalid("--rdma applies to --listen"));
        }
        self.data_address()?;
        Ok(())
    }
}

/// The result of parsing the command line.
#[allow(clippy::large_enum_variant)]
pub enum Parsed {
    Run(Config),
    Help,
}

pub fn help_text() -> String {
    "selene [options]\n\
     \n\
     --name NAME                     default \"default\"\n\
     --socket PATH                   default /run/selene/NAME.sock\n\
     --root PATH                     NVMe directory, owned by this manager alone\n\
     --memory BYTES                  shared arena, default 1G\n\
     --nvme BYTES                    NVMe budget; required with --root\n\
     --staging BYTES                 shared staging, default 4G\n\
     --listen ADDR:PORT              serve instances on other hosts\n\
     --listen-data ADDR:PORT         where they copy bytes, default --listen's next port\n\
     --tls-cert PATH --tls-key PATH  TLS for --listen (clients verify against their CA)\n\
     --rdma                          offer RDMA (experimental; not with TLS)\n\
     --metrics ADDR:PORT\n\
     \n\
     Sizes accept binary K, M, G and T suffixes."
        .to_string()
}

/// Parse a command line (without the program name).
pub fn parse_args<I>(args: I) -> io::Result<Parsed>
where
    I: IntoIterator<Item = String>,
{
    let args: Vec<String> = args.into_iter().collect();
    for argument in &args {
        if argument == "--help" || argument == "-h" {
            return Ok(Parsed::Help);
        }
    }

    let mut config = Config::defaults();
    let mut explicit_socket = false;

    let mut index = 0;
    while index < args.len() {
        let option = args[index].as_str();
        let value = |index: usize| -> io::Result<&str> {
            args.get(index + 1)
                .map(String::as_str)
                .ok_or_else(|| invalid(format!("missing value for {option}")))
        };
        match option {
            "--name" => {
                config.name = value(index)?.to_string();
                index += 2;
            }
            "--socket" => {
                config.socket = value(index)?.into();
                config.socket_explicit = true;
                explicit_socket = true;
                index += 2;
            }
            "--root" => {
                config.root = Some(value(index)?.into());
                index += 2;
            }
            "--memory" => {
                config.memory = parse_size(value(index)?)?;
                index += 2;
            }
            "--nvme" => {
                config.nvme = parse_size(value(index)?)?;
                index += 2;
            }
            "--staging" => {
                config.staging = parse_size(value(index)?)?;
                index += 2;
            }
            "--listen" => {
                config.listen = Some(parse_address(value(index)?)?);
                index += 2;
            }
            "--listen-data" => {
                config.listen_data = Some(parse_address(value(index)?)?);
                index += 2;
            }
            "--tls-cert" => {
                config.tls_cert = Some(value(index)?.into());
                index += 2;
            }
            "--tls-key" => {
                config.tls_key = Some(value(index)?.into());
                index += 2;
            }
            "--rdma" => {
                config.rdma = true;
                index += 1;
            }
            "--metrics" => {
                config.metrics = Some(parse_address(value(index)?)?);
                index += 2;
            }
            other => {
                return Err(invalid(format!("unknown option {other}")));
            }
        }
    }
    if !explicit_socket && !config.socket_explicit {
        config.socket = Config::default_socket(&config.name);
    }
    config.validate()?;
    Ok(Parsed::Run(config))
}

/// Parse a byte size with an optional binary K, M, G or T suffix.
pub fn parse_size(text: &str) -> io::Result<u64> {
    let (number, multiplier) = match text.as_bytes().last() {
        Some(b'K' | b'k') => (&text[..text.len() - 1], 1u64 << 10),
        Some(b'M' | b'm') => (&text[..text.len() - 1], 1u64 << 20),
        Some(b'G' | b'g') => (&text[..text.len() - 1], 1u64 << 30),
        Some(b'T' | b't') => (&text[..text.len() - 1], 1u64 << 40),
        _ => (text, 1),
    };
    if number.is_empty() {
        return Err(invalid(format!("invalid byte size {text:?}")));
    }
    number
        .parse::<u64>()
        .ok()
        .and_then(|value| value.checked_mul(multiplier))
        .ok_or_else(|| invalid(format!("invalid byte size {text:?}")))
}

fn parse_address(text: &str) -> io::Result<SocketAddr> {
    text.parse()
        .map_err(|_| invalid(format!("invalid address {text:?}")))
}

/// A manager name is also a directory name.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

pub fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    fn run(values: &[&str]) -> Config {
        match parse_args(args(values)).unwrap() {
            Parsed::Run(config) => config,
            Parsed::Help => panic!("unexpected help"),
        }
    }

    #[test]
    fn defaults_match_the_spec() {
        let config = run(&[]);
        assert_eq!(config.name, "default");
        assert_eq!(config.socket, PathBuf::from("/run/selene/default.sock"));
        assert_eq!(config.memory, 1 << 30);
        assert_eq!(config.staging, 4 << 30);
        assert!(config.listen.is_none() && !config.rdma);
    }

    #[test]
    fn size_suffixes_are_binary() {
        assert_eq!(parse_size("1K").unwrap(), 1024);
        assert_eq!(parse_size("2M").unwrap(), 2 << 20);
        assert_eq!(parse_size("3G").unwrap(), 3 << 30);
        assert_eq!(parse_size("1T").unwrap(), 1 << 40);
        assert_eq!(parse_size("4096").unwrap(), 4096);
        assert!(parse_size("").is_err());
        assert!(parse_size("1X").is_err());
        assert!(parse_size("K").is_err());
    }

    #[test]
    fn socket_default_follows_name_unless_set() {
        let config = run(&["--name", "pool-a"]);
        assert_eq!(config.socket, PathBuf::from("/run/selene/pool-a.sock"));
        let config = run(&["--name", "pool-a", "--socket", "/tmp/x.sock"]);
        assert_eq!(config.socket, PathBuf::from("/tmp/x.sock"));
    }

    #[test]
    fn flags_parse_and_validate() {
        let config = run(&[
            "--name",
            "pool",
            "--root",
            "/var/lib/selene",
            "--nvme",
            "1G",
            "--memory",
            "2G",
            "--staging",
            "16M",
        ]);
        assert_eq!(config.nvme, 1 << 30);
        assert_eq!(config.memory, 2 << 30);
        assert_eq!(config.staging, 16 << 20);
    }

    #[test]
    fn nvme_and_root_go_together() {
        assert!(parse_args(args(&["--nvme", "1G"])).is_err());
        assert!(parse_args(args(&["--root", "/r", "--nvme", "4096"])).is_err());
        assert!(parse_args(args(&["--root", "/r", "--nvme", "1G"])).is_ok());
    }

    #[test]
    fn unknown_options_and_missing_values_are_errors() {
        assert!(parse_args(args(&["--bogus"])).is_err());
        assert!(parse_args(args(&["--memory"])).is_err());
        // The peer options of the first protocol are gone.
        assert!(parse_args(args(&["--listen-peer", "127.0.0.1:1"])).is_err());
        assert!(parse_args(args(&["--upstream", "a=b"])).is_err());
    }

    #[test]
    fn tls_goes_with_listen_and_in_pairs() {
        assert!(parse_args(args(&["--tls-cert", "/c", "--tls-key", "/k"])).is_err());
        assert!(
            parse_args(args(&["--listen", "127.0.0.1:9000", "--tls-cert", "/c"])).is_err(),
            "a certificate needs its key"
        );
        let config = run(&[
            "--listen",
            "127.0.0.1:9000",
            "--tls-cert",
            "/c",
            "--tls-key",
            "/k",
        ]);
        assert!(config.listen.is_some() && config.tls_cert.is_some());
    }

    #[test]
    fn network_listening_is_named_in_a_startup_warning() {
        let mut config = Config::defaults();
        assert!(config.startup_warnings().is_empty());
        config.listen = Some("127.0.0.1:9200".parse().unwrap());
        let warnings = config.startup_warnings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("network listening enabled"));
        assert!(warnings[0].contains("127.0.0.1:9200"));
        assert!(warnings[0].contains("plaintext"));
        config.tls_cert = Some("/c".into());
        config.tls_key = Some("/k".into());
        assert!(config.startup_warnings()[0].contains("TLS"));
    }

    #[test]
    fn the_data_listener_follows_the_session_listener_unless_told_otherwise() {
        let mut config = Config::defaults();
        assert_eq!(config.data_address().unwrap(), None);
        config.listen = Some("10.0.0.5:9200".parse().unwrap());
        assert_eq!(
            config.data_address().unwrap(),
            Some("10.0.0.5:9201".parse().unwrap())
        );
        config.listen_data = Some("10.0.0.5:7000".parse().unwrap());
        assert_eq!(
            config.data_address().unwrap(),
            Some("10.0.0.5:7000".parse().unwrap())
        );
        // Port 0 means any free port for both.
        config.listen = Some("127.0.0.1:0".parse().unwrap());
        config.listen_data = None;
        assert_eq!(
            config.data_address().unwrap(),
            Some("127.0.0.1:0".parse().unwrap())
        );
        // The last port has no next one, and a data port alone names nothing to serve.
        config.listen = Some("127.0.0.1:65535".parse().unwrap());
        assert!(config.validate().is_err());
        config.listen = None;
        config.listen_data = Some("127.0.0.1:7000".parse().unwrap());
        assert!(config.validate().is_err());
        let parsed = run(&[
            "--listen",
            "127.0.0.1:9000",
            "--listen-data",
            "127.0.0.1:9100",
        ]);
        assert_eq!(parsed.listen_data, Some("127.0.0.1:9100".parse().unwrap()));
        let warnings = parsed.startup_warnings();
        assert!(
            warnings[0].contains("data on 127.0.0.1:9100"),
            "{warnings:?}"
        );
    }

    #[test]
    fn rdma_is_experimental_and_never_combined_with_tls() {
        let mut config = Config::defaults();
        config.listen = Some("127.0.0.1:9200".parse().unwrap());
        config.rdma = true;
        if cfg!(feature = "rdma") {
            config.validate().unwrap();
            let warnings = config.startup_warnings();
            assert!(warnings.contains(&"selene: warning: RDMA enabled (experimental)".to_string()));
            config.tls_cert = Some("/c".into());
            config.tls_key = Some("/k".into());
            let error = config.validate().unwrap_err().to_string();
            assert!(error.contains("not encrypted"), "{error}");
        } else {
            let error = config.validate().unwrap_err().to_string();
            assert!(error.contains("rdma"), "{error}");
        }
    }
}
