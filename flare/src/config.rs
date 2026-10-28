//! Host configuration defaults for native serving and prepared-model acquisition.

use std::{
    collections::HashMap,
    fs,
    net::IpAddr,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use clap::{ArgMatches, parser::ValueSource};
use flare_helios::{
    offload::{parse_drain_on_shutdown, parse_max_age, parse_size},
    runtime_package::MemoryLimit,
};
use serde::Deserialize;

use crate::cli;

type Failure = Box<dyn std::error::Error + Send + Sync>;

#[derive(Default, Deserialize)]
#[serde(default)]
pub(crate) struct Configuration {
    server: Server,
    defaults: Model,
    models: HashMap<String, Model>,
    memory: Memory,
    prefix_cache: PrefixCache,
    paths: Paths,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Server {
    host: Option<IpAddr>,
    port: Option<std::num::NonZeroU16>,
    api_key: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Model {
    max_concurrency: Option<NonZeroUsize>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Memory {
    reserve: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PrefixCache {
    memory: Option<String>,
    disk_limit: Option<String>,
    drain_on_shutdown: Option<String>,
    max_age: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Paths {
    package_cache: Option<String>,
    prefix_cache: Option<String>,
}

impl Configuration {
    pub(crate) fn load(path: &Path, required: bool) -> Result<Self, Failure> {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if !required && error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => return Err(format!("cannot read {}: {error}", path.display()).into()),
        };
        Self::parse(&text).map_err(|error| format!("{}: {error}", path.display()).into())
    }

    fn parse(text: &str) -> Result<Self, Failure> {
        let configuration: Self = toml::from_str(text)?;
        size(configuration.memory.reserve.as_deref(), "memory.reserve")?;
        size(
            nonempty(configuration.prefix_cache.memory.as_deref()),
            "prefix_cache.memory",
        )?;
        size(
            nonempty(configuration.prefix_cache.disk_limit.as_deref()),
            "prefix_cache.disk_limit",
        )?;
        if let Some(value) = nonempty(configuration.prefix_cache.drain_on_shutdown.as_deref()) {
            parse_drain_on_shutdown(value)
                .map_err(|error| format!("prefix_cache.drain_on_shutdown: {error}"))?;
        }
        if let Some(value) = nonempty(configuration.prefix_cache.max_age.as_deref()) {
            parse_max_age(value).map_err(|error| format!("prefix_cache.max_age: {error}"))?;
        }
        Ok(configuration)
    }

    pub(crate) fn package_cache(&self, home: &Path) -> Option<PathBuf> {
        nonempty(self.paths.package_cache.as_deref()).map(|path| expand(home, Path::new(path)))
    }

    pub(crate) fn apply(
        &self,
        cli: &mut cli::Cli,
        matches: &ArgMatches,
        home: &Path,
    ) -> Result<(), Failure> {
        let cache = self
            .package_cache(home)
            .unwrap_or_else(|| home.join(".cache/flare/packages"));
        match &mut cli.command {
            cli::Command::Run(run) => {
                let matches = matches.subcommand_matches("run").expect("run matches");
                if matches.value_source("host") != Some(ValueSource::CommandLine) {
                    if let Some(host) = self.server.host {
                        run.host = host;
                    }
                }
                if matches.value_source("port") != Some(ValueSource::CommandLine) {
                    if let Some(port) = self.server.port {
                        run.port = port.get();
                    }
                }
                let serving = &mut run.serving;
                if serving.memory_limit.is_none() {
                    serving.memory_limit = size(self.memory.reserve.as_deref(), "memory.reserve")?
                        .map(MemoryLimit::Reserve);
                }
                if serving.prefix_cache_memory.is_none() {
                    serving.prefix_cache_memory = size(
                        nonempty(self.prefix_cache.memory.as_deref()),
                        "prefix_cache.memory",
                    )?;
                }
                if serving.prefix_cache_disk_limit.is_none() {
                    serving.prefix_cache_disk_limit = size(
                        nonempty(self.prefix_cache.disk_limit.as_deref()),
                        "prefix_cache.disk_limit",
                    )?;
                }
                if serving.prefix_cache_drain_on_shutdown.is_none() {
                    serving.prefix_cache_drain_on_shutdown =
                        nonempty(self.prefix_cache.drain_on_shutdown.as_deref())
                            .map(parse_drain_on_shutdown)
                            .transpose()?;
                }
                if serving.prefix_cache_max_age.is_none() {
                    serving.prefix_cache_max_age = nonempty(self.prefix_cache.max_age.as_deref())
                        .map(parse_max_age)
                        .transpose()?;
                }
                if serving.prefix_cache_disk.is_none() {
                    serving.prefix_cache_disk = nonempty(self.paths.prefix_cache.as_deref())
                        .map(|path| expand(home, Path::new(path)));
                }
                serving
                    .hub
                    .model_cache
                    .get_or_insert_with(|| run.package_cache.clone().unwrap_or(cache));
            }
            cli::Command::Download(args) => {
                args.hub.model_cache.get_or_insert(cache);
            }
            cli::Command::Inspect(args) => {
                args.model_cache.get_or_insert(cache);
            }
            cli::Command::Models(args) => {
                let cache = args.model_cache.get_or_insert(cache).clone();
                match &mut args.command {
                    cli::CacheCommand::Inspect(inspect) => {
                        inspect.model_cache.get_or_insert(cache);
                    }
                    cli::CacheCommand::Download(download) => {
                        download.hub.model_cache.get_or_insert(cache);
                    }
                    _ => {}
                }
            }
            cli::Command::Orbit(args) => match &mut args.command {
                cli::OrbitCommand::Run(run) => {
                    run.serving.hub.model_cache.get_or_insert(cache);
                }
                cli::OrbitCommand::Push(push) => {
                    push.serving.hub.model_cache.get_or_insert(cache);
                }
                _ => {}
            },
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn apply_model(
        &self,
        serving: &mut cli::ServingOptions,
        names: &[&str],
    ) -> Result<(), Failure> {
        if serving.max_concurrency.is_some() {
            return Ok(());
        }
        let mut selected = None;
        for name in names {
            if let Some(value) = self
                .models
                .get(*name)
                .and_then(|model| model.max_concurrency)
            {
                if selected.is_some_and(|previous| previous != value) {
                    return Err("conflicting max_concurrency settings for the package's model ID and aliases".into());
                }
                selected = Some(value);
            }
        }
        serving.max_concurrency = selected.or(self.defaults.max_concurrency);
        Ok(())
    }

    pub(crate) fn api_key(&self, file: Option<&Path>) -> Result<String, Failure> {
        match file {
            Some(path) => fs::read_to_string(path)
                .map(|value| value.trim().to_owned())
                .map_err(|error| {
                    format!("cannot read API key file {}: {error}", path.display()).into()
                }),
            None => Ok(self.server.api_key.clone().unwrap_or_default()),
        }
    }
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.trim().is_empty())
}

fn size(value: Option<&str>, key: &str) -> Result<Option<u64>, Failure> {
    value
        .map(|value| parse_size(value).map_err(|error| format!("{key}: {error}").into()))
        .transpose()
}

pub(crate) fn expand(home: &Path, path: &Path) -> PathBuf {
    match path.strip_prefix("~") {
        Ok(rest) => home.join(rest),
        Err(_) => path.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, FromArgMatches, Parser};

    fn configured(text: &str, arguments: &[&str], names: &[&str]) -> (Configuration, cli::Run) {
        let config = Configuration::parse(text).unwrap();
        let matches = cli::Cli::command().try_get_matches_from(arguments).unwrap();
        let mut cli = cli::Cli::from_arg_matches(&matches).unwrap();
        config
            .apply(&mut cli, &matches, Path::new("/home/test"))
            .unwrap();
        let cli::Command::Run(mut run) = cli.command else {
            panic!()
        };
        config.apply_model(&mut run.serving, names).unwrap();
        (config, *run)
    }

    #[test]
    fn prefix_cache_drain_on_shutdown_defaults_and_flag_precedence() {
        let args = ["flare", "run", "model.safetensors"];
        let (_, run) = configured("", &args, &[]);
        assert_eq!(
            run.load_options().prefix_cache.drain_on_shutdown,
            std::time::Duration::ZERO
        );
        for value in ["on", "90s", "off", "0"] {
            let text = format!("[prefix_cache]\ndrain_on_shutdown='{value}'");
            let (_, run) = configured(&text, &args, &[]);
            assert_eq!(
                run.load_options().prefix_cache.drain_on_shutdown,
                parse_drain_on_shutdown(value).unwrap()
            );
            let (_, run) = configured(
                &text,
                &[
                    "flare",
                    "run",
                    "model.safetensors",
                    "--prefix-cache-drain-on-shutdown=0.5s",
                ],
                &[],
            );
            assert_eq!(
                run.load_options().prefix_cache.drain_on_shutdown,
                std::time::Duration::from_millis(500)
            );
        }
        let (_, run) = configured(
            "[prefix_cache]\ndrain_on_shutdown='off'",
            &[
                "flare",
                "run",
                "model.safetensors",
                "--prefix-cache-drain-on-shutdown",
            ],
            &[],
        );
        assert_eq!(
            run.load_options().prefix_cache.drain_on_shutdown,
            std::time::Duration::MAX
        );
        let (_, run) = configured(
            "[prefix_cache]\ndrain_on_shutdown='on'",
            &[
                "flare",
                "run",
                "model.safetensors",
                "--prefix-cache-drain-on-shutdown=off",
            ],
            &[],
        );
        assert_eq!(
            run.load_options().prefix_cache.drain_on_shutdown,
            std::time::Duration::ZERO
        );
        for value in ["on", "off", "1ms", "2h", "0.0001s"] {
            let run = cli::Cli::try_parse_from([
                "flare",
                "run",
                "model.safetensors",
                &format!("--prefix-cache-drain-on-shutdown={value}"),
            ])
            .unwrap();
            let cli::Command::Run(run) = run.command else {
                panic!()
            };
            let forwarded = run.rank_options();
            let mut args = vec!["flare", "run", "model.safetensors"];
            args.extend(forwarded.iter().map(String::as_str));
            let peer = cli::Cli::try_parse_from(args).unwrap();
            let cli::Command::Run(peer) = peer.command else {
                panic!()
            };
            assert_eq!(
                run.load_options().prefix_cache.drain_on_shutdown,
                peer.load_options().prefix_cache.drain_on_shutdown
            );
        }
        for invalid in ["-1", "NaN", "inf", "1e30h", "later", "0s"] {
            assert!(
                Configuration::parse(&format!("[prefix_cache]\ndrain_on_shutdown='{invalid}'"))
                    .is_err()
            );
            assert!(
                cli::Cli::try_parse_from([
                    "flare",
                    "run",
                    "model.safetensors",
                    &format!("--prefix-cache-drain-on-shutdown={invalid}")
                ])
                .is_err()
            );
        }
    }

    const SETTINGS: &str = r#"
[server]
host = "::1"
port = 9001
api_key = "configured-secret"
[memory]
reserve = "2G"
[defaults]
max_concurrency = 8
model = "preparation-recipe"
image_concurrency = 2
speculative_token_capacity = 128
speculative_graph_budget = 64
[models."recipe"]
max_concurrency = 4
[prefix_cache]
memory = "1.5G"
disk_limit = "20GiB"
[paths]
package_cache = "~/packages"
prefix_cache = "~/prefix"
hub_cache = "/checkpoint-cache"
kernel_cache = "/kernels"
export_staging = "/staging"
cuda = "/opt/cuda"
[preparation]
compile_workers = 4
autotune_effort = "off"
[publishing]
namespace = "publisher"
"#;

    #[test]
    fn runtime_defaults_reach_bind_load_options_and_authentication() {
        let (config, run) = configured(
            SETTINGS,
            &["flare", "run", "model"],
            &["org/model", "recipe"],
        );
        assert_eq!(run.bind_address().to_string(), "[::1]:9001");
        assert_eq!(config.api_key(None).unwrap(), "configured-secret");
        assert_eq!(run.memory_limit, Some(MemoryLimit::Reserve(2 << 30)));
        assert_eq!(run.max_concurrency.unwrap().get(), 4);
        assert_eq!(run.prefix_cache_memory, Some(3 << 29));
        assert_eq!(run.prefix_cache_disk_limit, Some(20 << 30));
        assert_eq!(
            run.prefix_cache_disk.as_deref(),
            Some(Path::new("/home/test/prefix"))
        );
        assert_eq!(
            run.hub.model_cache.as_deref(),
            Some(Path::new("/home/test/packages"))
        );
        let options = run.load_options();
        assert_eq!(options.max_concurrency, Some(4));
        assert_eq!(options.memory_limit, run.memory_limit);
        assert!(run.mtp_max_verification_token_budget.is_none());
        assert!(run.mtp_verification_token_budget.is_none());
    }

    #[test]
    fn explicit_flags_win_even_when_equal_to_builtins_or_zero() {
        let (_, run) = configured(
            SETTINGS,
            &[
                "flare",
                "run",
                "model",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--memory-limit",
                "0.8",
                "--max-concurrency",
                "2",
                "--prefix-cache-memory",
                "0",
                "--prefix-cache-disk-limit",
                "0",
                "--prefix-cache-disk",
                "/explicit-prefix",
                "--model-cache",
                "/explicit-cache",
            ],
            &["recipe"],
        );
        assert_eq!(run.bind_address().to_string(), "127.0.0.1:8000");
        assert_eq!(run.memory_limit, Some(MemoryLimit::Fraction(0.8)));
        assert_eq!(run.max_concurrency.unwrap().get(), 2);
        assert_eq!(run.prefix_cache_memory, Some(0));
        assert_eq!(run.prefix_cache_disk_limit, Some(0));
        assert_eq!(
            run.prefix_cache_disk.as_deref(),
            Some(Path::new("/explicit-prefix"))
        );
        assert_eq!(
            run.hub.model_cache.as_deref(),
            Some(Path::new("/explicit-cache"))
        );
    }

    #[test]
    fn empty_configuration_preserves_package_defaults() {
        let (config, run) = configured("", &["flare", "run", "model"], &["recipe"]);
        assert!(run.memory_limit.is_none());
        assert!(run.max_concurrency.is_none());
        assert!(run.prefix_cache_disk.is_none());
        assert!(run.prefix_cache_memory.is_none());
        assert!(run.prefix_cache_disk_limit.is_none());
        assert_eq!(run.bind_address().to_string(), "127.0.0.1:8000");
        assert_eq!(config.api_key(None).unwrap(), "");
    }

    #[test]
    fn model_overrides_use_signed_names_and_detect_conflicts() {
        let (_, run) = configured(SETTINGS, &["flare", "run", "recipe"], &["different-model"]);
        assert_eq!(
            run.max_concurrency.unwrap().get(),
            8,
            "filename must not select model settings"
        );
        let (_, run) = configured(
            SETTINGS,
            &[
                "flare",
                "run",
                "model",
                "--served-model-name",
                "different-model",
            ],
            &["recipe"],
        );
        assert_eq!(
            run.max_concurrency.unwrap().get(),
            4,
            "HTTP alias override must not select configuration"
        );
        let config =
            Configuration::parse("[models.a]\nmax_concurrency=2\n[models.b]\nmax_concurrency=3")
                .unwrap();
        let cli::Command::Run(mut run) = cli::Cli::parse_from(["flare", "run", "model"]).command
        else {
            panic!()
        };
        assert!(config.apply_model(&mut run.serving, &["a", "b"]).is_err());
        run.serving.max_concurrency = NonZeroUsize::new(1);
        config.apply_model(&mut run.serving, &["a", "b"]).unwrap();
        assert_eq!(run.max_concurrency.unwrap().get(), 1);
    }

    #[test]
    fn api_key_file_overrides_configuration_including_empty_files() {
        let config = Configuration::parse(SETTINGS).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key");
        fs::write(&path, " file-secret\n").unwrap();
        assert_eq!(config.api_key(Some(&path)).unwrap(), "file-secret");
        fs::write(&path, "").unwrap();
        assert_eq!(config.api_key(Some(&path)).unwrap(), "");
        fs::remove_file(&path).unwrap();
        assert!(config.api_key(Some(&path)).is_err());
    }

    #[test]
    fn missing_default_is_optional_but_explicit_and_malformed_files_fail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        assert!(Configuration::load(&path, false).is_ok());
        assert!(Configuration::load(&path, true).is_err());
        fs::write(&path, "[server]\nport = 'bad'").unwrap();
        assert!(
            Configuration::load(&path, false)
                .err()
                .unwrap()
                .to_string()
                .contains("config.toml")
        );
    }

    #[test]
    fn invalid_runtime_settings_fail_before_loading_a_package() {
        for text in [
            "[server]\nhost='not-an-ip'",
            "[server]\nport=0",
            "[server]\nport=65536",
            "[server]\napi_key=7",
            "[defaults]\nmax_concurrency=0",
            "[models.recipe]\nmax_concurrency=-1",
            "[memory]\nreserve='-1G'",
            "[memory]\nreserve=''",
            "[prefix_cache]\nmemory='many'",
            "[prefix_cache]\ndisk_limit='-1'",
            "[prefix_cache]\ndisk_transport='invalid'",
            "[prefix_cache]\ndisk_transport='staged'",
            "[prefix_cache]\ndisk_transport='direct'",
            "[paths]\npackage_cache=12",
        ] {
            assert!(Configuration::parse(text).is_err(), "{text}");
        }
    }

    #[test]
    fn configured_rank_options_round_trip_including_zero_reserve() {
        for text in [
            SETTINGS.to_owned(),
            SETTINGS.replace("reserve = \"2G\"", "reserve = \"0\""),
        ] {
            let (_, root) = configured(&text, &["flare", "run", "model"], &["recipe"]);
            let mut argv = vec![
                "flare".to_owned(),
                "run".into(),
                "--rank".into(),
                "1".into(),
            ];
            argv.extend(root.rank_options());
            let cli::Command::Run(peer) = cli::Cli::try_parse_from(argv).unwrap().command else {
                panic!()
            };
            assert_eq!(peer.memory_limit, root.memory_limit);
            assert_eq!(peer.max_concurrency, root.max_concurrency);
            assert_eq!(peer.prefix_cache_memory, root.prefix_cache_memory);
            assert_eq!(peer.prefix_cache_disk, root.prefix_cache_disk);
            assert_eq!(peer.prefix_cache_disk_limit, root.prefix_cache_disk_limit);
        }
    }

    #[test]
    fn cache_commands_and_download_use_configuration_and_explicit_overrides() {
        for arguments in [
            vec!["flare", "models", "list"],
            vec!["flare", "download", "org/model"],
            vec!["flare", "models", "download", "org/model"],
            vec!["flare", "inspect", "org/model@version"],
            vec!["flare", "models", "inspect", "org/model@version"],
        ] {
            for explicit in [false, true] {
                let mut arguments = arguments.clone();
                if explicit {
                    arguments.extend(["--model-cache", "/explicit"]);
                }
                let matches = cli::Cli::command().try_get_matches_from(arguments).unwrap();
                let mut cli = cli::Cli::from_arg_matches(&matches).unwrap();
                Configuration::parse(SETTINGS)
                    .unwrap()
                    .apply(&mut cli, &matches, Path::new("/home/test"))
                    .unwrap();
                let cache = match cli.command {
                    cli::Command::Models(args) => match args.command {
                        cli::CacheCommand::Inspect(inspect) => inspect.model_cache,
                        cli::CacheCommand::Download(download) => download.hub.model_cache,
                        _ => args.model_cache,
                    },
                    cli::Command::Inspect(args) => args.model_cache,
                    cli::Command::Download(args) => args.hub.model_cache,
                    _ => panic!(),
                };
                assert_eq!(
                    cache.unwrap(),
                    PathBuf::from(if explicit {
                        "/explicit"
                    } else {
                        "/home/test/packages"
                    })
                );
            }
        }
        let (_, run) = configured(
            SETTINGS,
            &["flare", "run", "model", "--package-cache", "/fallback"],
            &[],
        );
        assert_eq!(run.hub.model_cache.as_deref(), Some(Path::new("/fallback")));
    }

    #[test]
    fn custom_config_flag_is_global_and_blank_prefix_values_remain_unspecified() {
        for args in [
            vec!["flare", "--config", "custom.toml", "run", "model"],
            vec!["flare", "run", "model", "--config", "custom.toml"],
        ] {
            assert_eq!(
                cli::Cli::try_parse_from(args).unwrap().config,
                Some(PathBuf::from("custom.toml"))
            );
        }
        let (_, run) = configured(
            "[prefix_cache]\nmemory=''\ndisk_limit=''\n[paths]\nprefix_cache=''",
            &["flare", "run", "model"],
            &[],
        );
        assert!(run.prefix_cache_memory.is_none());
        assert!(run.prefix_cache_disk_limit.is_none());
        assert!(run.prefix_cache_disk.is_none());
    }
    #[test]
    fn prefix_cache_age_reaches_flags_config_and_ranks() {
        let (_, run) = configured("", &["flare", "run", "model.safetensors"], &[]);
        assert_eq!(
            run.load_options().prefix_cache.max_age,
            std::time::Duration::ZERO
        );
        for value in ["off", "2h", "0.5s"] {
            let (_, run) = configured(
                &format!("[prefix_cache]\nmax_age='{value}'"),
                &["flare", "run", "model.safetensors"],
                &[],
            );
            assert_eq!(
                run.load_options().prefix_cache.max_age,
                parse_max_age(value).unwrap()
            );
            let mut args = vec![
                "flare".to_string(),
                "run".into(),
                "model.safetensors".into(),
            ];
            args.extend(run.rank_options());
            let cli::Command::Run(peer) = cli::Cli::try_parse_from(args).unwrap().command else {
                panic!()
            };
            assert_eq!(
                peer.load_options().prefix_cache.max_age,
                run.load_options().prefix_cache.max_age
            );
        }
        for invalid in ["on", "0", "-1s", "NaN", "2d"] {
            assert!(Configuration::parse(&format!("[prefix_cache]\nmax_age='{invalid}'")).is_err());
        }
    }
}
