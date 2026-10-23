//! Serve a prepared model package without a Python interpreter.

#[cfg(unix)]
mod agent_client;
mod cli;
mod cluster;
#[cfg(unix)]
mod distribution;
mod environment;
#[cfg(unix)]
mod fabric;
#[cfg(unix)]
mod fabric_transport;
mod hardware;
mod hub;
mod identity;
mod inspect;
#[cfg(unix)]
mod launch;
#[cfg(unix)]
mod orbit;
#[cfg(unix)]
mod package_cache;
mod trust;
#[cfg(windows)]
mod windows;

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use clap::{CommandFactory, FromArgMatches};
use flare_helios::{
    runtime_package::NativeModel,
    server::{Bootstrap, Server},
    startup::StartupProgress,
};

static STOP: AtomicBool = AtomicBool::new(false);
#[cfg(unix)]
extern "C" fn stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = cli::Cli::command().get_matches();
    let parsed = cli::Cli::from_arg_matches(&matches).unwrap_or_else(|error| error.exit());
    let mut args = match parsed.command {
        cli::Command::Run(args) => *args,
        cli::Command::Download(args) => {
            environment::configure(&args.gpus)?;
            let path = hub::download(&args.repository, &args.hub, args.gpus.len())?;
            println!("{}", path.display());
            return Ok(());
        }
        cli::Command::Inspect(args) => return inspect::run(args),
        cli::Command::HardwareRevision { gpu_count } => {
            let devices: Vec<_> = (0..gpu_count).collect();
            environment::configure(&devices)?;
            println!("{}", hardware::detect(gpu_count as usize)?);
            return Ok(());
        }
        #[cfg(unix)]
        cli::Command::CheckpointTransport { socket } => {
            return fabric_transport::run(&socket).map_err(|e| e.to_string().into());
        }
        #[cfg(windows)]
        cli::Command::Agent(_) => {
            return Err("multi-host serving is not supported on Windows".into());
        }
        #[cfg(windows)]
        cli::Command::Orbit(_) => {
            return Err("orbit is not supported on Windows".into());
        }
        #[cfg(unix)]
        cli::Command::Agent(agent_args) => {
            return match agent_args.command {
                None => orbit::agent::serve(
                    &agent_args.listen,
                    orbit::store::AgentDir::new(orbit::store::AgentDir::default_root()),
                    orbit::cache::default_root(),
                ),
                Some(cli::AgentCommand::Install {
                    join_code,
                    accept_first,
                    listen,
                }) => orbit::agent::install(&listen, join_code.as_deref(), accept_first),
                Some(cli::AgentCommand::Pair {
                    hosts,
                    code,
                    agents_file,
                    controller_key,
                }) => agent_client::Client::open(agents_file.as_deref(), controller_key.as_deref())
                    .and_then(|mut client| client.pair(&hosts, code.as_deref())),
                Some(cli::AgentCommand::Reset) => orbit::agent::reset(),
            }
            .map_err(|e| e.to_string().into());
        }
        #[cfg(unix)]
        cli::Command::Orbit(orbit_args) => {
            return orbit_command(orbit_args, matches.subcommand_matches("orbit").unwrap())
                .map_err(|e| e.to_string().into());
        }
        cli::Command::Trust {
            command:
                cli::TrustCommand::Revoke {
                    fingerprint,
                    trust_store,
                },
        } => {
            return trust::revoke(&fingerprint, trust_store.as_deref());
        }
    };
    #[cfg(windows)]
    if args.rank.is_some()
        || args.rendezvous.is_some()
        || !args.hosts.is_empty()
        || args.roce_ring
        || args.roce_chain
    {
        return Err("multi-host serving is not supported on Windows".into());
    }
    #[cfg(windows)]
    unsafe {
        if windows_sys::Win32::System::Console::SetConsoleCtrlHandler(Some(stop_console), 1) == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    let bind = args.bind_address();
    if args.rank.is_some_and(|rank| rank != 0) {
        args.serving.select_gpus(1);
    }
    if !args.gpus.is_empty() {
        environment::configure(&args.gpus)?;
    }
    let key = args
        .api_key_file
        .as_ref()
        .map(|path| {
            std::fs::read_to_string(path)
                .map(|value| value.trim().to_owned())
                .map_err(|error| format!("cannot read API key file {}: {error}", path.display()))
        })
        .transpose()?;
    let mut progress = StartupProgress::stderr();
    #[cfg(unix)]
    let mut launched = None;
    #[cfg(unix)]
    let _lease;
    #[cfg(unix)]
    let (authorized, mut membership) = if args.rank.is_some_and(|rank| rank != 0) {
        if !args.hosts.is_empty() {
            return Err("--hosts applies only to rank zero".into());
        }
        let received = distribution::peer(&mut args, &mut progress).map_err(|e| e.to_string())?;
        _lease = received.lease;
        (received.authorized, Some(received.membership))
    } else {
        let (root, authorized, publisher) = load_package(&args, &mut progress)?;
        configure_package_devices(&mut args, &authorized, &mut progress)?;
        _lease = None;
        if !args.hosts.is_empty() {
            launched = Some(
                launch::Peers::start(&mut args, &authorized, &publisher, &root)
                    .map_err(|e| e.to_string())?,
            );
        }
        let membership = match (
            flare_helios::runtime_package::package_cluster(&authorized),
            args.rendezvous.as_ref(),
        ) {
            (Some(_), Some(_)) => {
                args.rank = Some(0);
                Some(
                    distribution::root(&authorized, &publisher, &root, &args, &progress)
                        .map_err(|e| e.to_string())?,
                )
            }
            (Some(_), None) => {
                return Err(
                    "a multi-host package requires --rendezvous HOST:PORT or --hosts".into(),
                );
            }
            (None, Some(_)) => {
                return Err("--rendezvous applies only to a multi-host package".into());
            }
            (None, None) => {
                if args.roce_ring || args.roce_chain {
                    return Err("--roce-ring and --roce-chain require a multi-host package".into());
                }
                None
            }
        };
        (authorized, membership)
    };
    #[cfg(windows)]
    let authorized = {
        let (_, authorized, _) = load_package(&args, &mut progress)?;
        configure_package_devices(&mut args, &authorized, &mut progress)?;
        authorized
    };
    #[cfg(windows)]
    let mut membership: Option<cluster::Membership> = None;
    if args.stage_only {
        #[cfg(unix)]
        if let Some(peers) = launched {
            peers.finish().map_err(|e| e.to_string())?;
        }
        return Ok(());
    }
    let options = args.load_options();
    let local_devices: Vec<i32> = (0..args.gpus.len() as i32).collect();
    let shape = cluster::Shape::from_flags(args.roce_ring, args.roce_chain);
    let rendezvous = args.rendezvous;
    // SAFETY: The trusted package owners remain live until every rank drains.
    let owner = unsafe {
        NativeModel::load_agreeing(
            authorized,
            &local_devices,
            &options,
            &progress,
            &mut |local| match membership.as_mut() {
                Some(membership) => membership.agree(local),
                None => Ok(local),
            },
        )?
    };
    let cluster = match (owner.cluster().cloned(), membership, rendezvous) {
        (Some(cluster), Some(membership), Some(address)) => Some((cluster, membership, address)),
        (None, None, None) => None,
        _ => return Err("the package's cluster differs from the rank that joined".into()),
    };
    let cluster = match cluster {
        Some((cluster, cluster::Membership::Peer(stream), address)) => {
            return serve_peer(owner, &cluster, stream, &address, progress, shape);
        }
        Some((cluster, cluster::Membership::Zero(streams), _)) => Some((cluster, streams)),
        None => None,
    };
    let phase = progress.phase("Read model configuration", None);
    let mut config = owner.server_config().clone();
    config["bind"] = bind.to_string().into();
    if let Some(alpha) = args.serving.mtp_adaptive_verification_alpha {
        config["mtp_adaptive_verification_alpha"] = alpha.into();
    }
    config["scheduler_policy"] = args.scheduler_policy.into();
    if let Some(name) = args.served_model_name {
        config["served_names"] = serde_json::json!([name]);
    }
    config["api_key"] = key.unwrap_or_default().into();
    phase.finish(config["model_id"].as_str().unwrap_or_default());
    serve(owner, config, progress, cluster, shape)?;
    #[cfg(unix)]
    if let Some(peers) = launched {
        peers.finish().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn configure_package_devices(
    args: &mut cli::Run,
    package: &flare_helios::runtime_package::AuthorizedPackage,
    progress: &mut StartupProgress,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.gpus.is_empty() {
        let local_ranks = if flare_helios::runtime_package::package_cluster(package).is_some() {
            1
        } else {
            flare_helios::runtime_package::rank_needs(package).len()
        };
        args.serving.select_gpus(local_ranks);
        progress.finish_animation();
        environment::configure(&args.gpus)?;
    }
    Ok(())
}

fn load_package(
    args: &cli::Run,
    progress: &mut StartupProgress,
) -> Result<
    (
        std::path::PathBuf,
        flare_helios::runtime_package::AuthorizedPackage,
        flare_helios::runtime_package::Publisher,
    ),
    Box<dyn std::error::Error>,
> {
    let package = args
        .package
        .as_ref()
        .ok_or("rank zero requires a package")?;
    let mut origin = None;
    let path = if hub::is_repository(package) {
        let text = package.to_str().ok_or("repository must be UTF-8")?;
        origin = Some(hub::repository(text)?.0.to_owned());
        hub::download(text, &args.hub, args.gpus.len().max(1))?
    } else {
        if args.hub.revision.is_some()
            || args.hub.filename.is_some()
            || args.hub.cache_dir.is_some()
        {
            return Err(
                "download options apply to Hugging Face repositories, not local package paths"
                    .into(),
            );
        }
        package.clone()
    };
    let root = path
        .canonicalize()
        .map_err(|e| format!("cannot access package {}: {e}", path.display()))?;
    let (authorized, publisher) = trust::load(&root, args, progress, origin.as_deref())?;
    Ok((root, authorized, publisher))
}

/// Run one `flare orbit` command as the controller on this machine.
#[cfg(unix)]
fn orbit_command(args: cli::OrbitArgs, matches: &clap::ArgMatches) -> Result<(), orbit::Failure> {
    let mut controller =
        orbit::controller::Controller::open(args.orbit_file.as_deref(), args.key_file.as_deref())?;
    match args.command {
        cli::OrbitCommand::Pair { hosts, code } => controller.pair(&hosts, code.as_deref()),
        cli::OrbitCommand::Controllers {
            command: cli::ControllersCommand::Add { name, key },
        } => controller.controllers_add(&name, &key),
        cli::OrbitCommand::Controllers {
            command: cli::ControllersCommand::Remove { key },
        } => controller.controllers_remove(&key),
        cli::OrbitCommand::Status { cluster } => controller.status(cluster.as_deref()),
        cli::OrbitCommand::Ls { cluster } => controller.ls(cluster.as_deref()),
        cli::OrbitCommand::Dashboard { cluster } => {
            orbit::dashboard::run(&controller, cluster.as_deref())
        }
        cli::OrbitCommand::Attach { cluster, group } => {
            controller.attach(cluster.as_deref(), group.as_deref())
        }
        cli::OrbitCommand::Stop { cluster, group } => {
            controller.stop(cluster.as_deref(), group.as_deref())
        }
        cli::OrbitCommand::Rm { group } => controller.remove(&group),
        cli::OrbitCommand::Forget { node } => controller.forget(&node),
        cli::OrbitCommand::Push(push) => controller.push(
            &push.package,
            &push.to,
            &mut push.run_args(),
            push.transfer_streams,
        ),
        cli::OrbitCommand::Packages { cluster } => controller.packages(cluster.as_deref()),
        cli::OrbitCommand::Run(run) => controller.run(
            run.package.as_deref(),
            run.cluster.as_deref(),
            orbit::controller::RunOptions {
                name: run.name.clone(),
                persistent: run.persistent,
                nodes: run.nodes.clone(),
                dry_run: run.dry_run,
                gpus: if matches
                    .subcommand_matches("run")
                    .unwrap()
                    .value_source("gpus")
                    != Some(clap::parser::ValueSource::CommandLine)
                {
                    Vec::new()
                } else {
                    run.serving.gpus.iter().map(|&g| g as u32).collect()
                },
                serving: run.rank_options(matches.subcommand_matches("run").unwrap()),
                rank_zero: run.rank_zero_options(),
                roce_ring: run.roce_ring,
                transfer_streams: run.transfer_streams,
                roce_chain: run.roce_chain,
                executable: run.executable.clone(),
                detach: run.detach,
                verify_weights: run.serving.verify_weights,
                run_args: run.run_args(),
            },
        ),
    }
}

/// Run a peer rank of a multi-host package: join rank zero and replay its coordinated steps until
/// rank zero drains every rank.
fn serve_peer(
    owner: NativeModel,
    cluster: &flare_helios::runtime_package::ClusterRank,
    stream: flare_helios::execution::RankStream,
    address: &str,
    mut progress: StartupProgress,
    shape: cluster::Shape,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut manifest = owner.manifests()?.remove(0);
    let mut joined = cluster::peer(&owner, cluster, stream, &progress, shape)?;
    let records = joined.records.take();
    manifest.collectives = Some(cluster::health(&joined.transport));
    progress.finish_animation();
    eprintln!(
        "Serving rank {} of {} for rank zero at {address}",
        cluster.rank, cluster.world
    );
    // SAFETY: The package owner and the transport stay live until rank zero confirms drain, and
    // are intentionally retained when it does not.
    let shutdown = unsafe {
        flare_helios::execution::serve_rank(manifest, joined.streams, cluster.rank, cluster.world)
    };
    if !shutdown.drained {
        std::mem::forget(records);
        std::mem::forget(joined.transport);
        std::mem::forget(owner);
        return Err(match shutdown.error {
            Some(error) => format!("native replay did not drain: {error}"),
            None => "native replay did not drain".into(),
        }
        .into());
    }
    // Rank zero closes its store after every rank drained; the service's transfers use this
    // rank's context and registered staging, which the package owner releases.
    if let Some(records) = records
        && let Err(error) = records.join()
    {
        // The service could not confirm its drain, so its transfers may still
        // use this rank's context and registered staging.
        std::mem::forget(joined.transport);
        std::mem::forget(owner);
        return Err(format!("host prefix record service did not drain: {error}").into());
    }
    // The proxy stops before the package releases the region it posts from.
    drop(joined.transport);
    drop(owner);
    match shutdown.error {
        Some(error) => Err(error.into()),
        None => Ok(()),
    }
}

fn serve(
    owner: NativeModel,
    config: serde_json::Value,
    mut progress: StartupProgress,
    cluster: Option<(
        flare_helios::runtime_package::ClusterRank,
        Vec<flare_helios::execution::RankStream>,
    )>,
    shape: cluster::Shape,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: This executable loads a trusted package and retains its resources
    // until the native server confirms every replay stream has drained.
    let phase = progress.phase("Load tokenizer", None);
    let mut manifests = owner.manifests()?;
    let mut manifest = manifests.remove(0);
    let mut transport = None;
    let peers = match cluster {
        Some((_, streams)) => {
            let joined = cluster::rank_zero(&owner, streams, &progress, shape)?;
            manifest.collectives = Some(cluster::health(&joined.transport));
            transport = Some(joined.transport);
            Some(flare_helios::execution::RankPeers::Processes(
                joined.streams,
            ))
        }
        None => (!manifests.is_empty())
            .then_some(flare_helios::execution::RankPeers::Threads(manifests)),
    };
    let bootstrap = Bootstrap::load(&config.to_string(), &manifest)?;
    phase.finish(format!(
        "{} token vocabulary",
        bootstrap.model.contract.vocabulary
    ));
    let phase = progress.phase("Start server", None);
    let context_tokens = bootstrap.model.limits.max_sequence_tokens;
    let server = match unsafe {
        Server::start(
            manifest,
            bootstrap.model,
            bootstrap.frontend,
            bootstrap.images,
            bootstrap.server,
            peers,
        )
    } {
        Ok(server) => server,
        Err(error) => {
            if !error.drained {
                std::mem::forget(transport);
                std::mem::forget(owner);
            }
            return Err(error.message.into());
        }
    };
    phase.finish(format!(
        "http://{} · {context_tokens} tokens/request",
        server.address()
    ));
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGINT, stop as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, stop as *const () as libc::sighandler_t);
    }
    progress.finish_animation();
    if let Err(error) = server.start_dashboard() {
        eprintln!("Could not start dashboard: {error}");
    }
    while !STOP.load(Ordering::Relaxed) && server.wait(Duration::from_millis(250)) {}
    server.stop_dashboard();
    // Allow the terminal multiplexer to finish restoring the main screen.
    std::thread::sleep(Duration::from_millis(50));
    let phase = progress.phase("Shut down server", None);
    let shutdown = server.close();
    if !shutdown.drained {
        std::mem::forget(transport);
        std::mem::forget(owner);
        return Err(match shutdown.error {
            Some(error) => format!("native replay did not drain: {error}"),
            None => "native replay did not drain".into(),
        }
        .into());
    }
    if let Some(error) = shutdown.error {
        return Err(error.into());
    }
    if !STOP.load(Ordering::Relaxed) {
        return Err("native server stopped unexpectedly".into());
    }
    drop(server);
    // The proxy stops before the package releases the region it posts from.
    drop(transport);
    drop(owner);
    phase.finish("");
    Ok(())
}

#[cfg(windows)]
unsafe extern "system" fn stop_console(event: u32) -> i32 {
    use windows_sys::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_C_EVENT};
    if matches!(event, CTRL_C_EVENT | CTRL_BREAK_EVENT) {
        STOP.store(true, Ordering::Relaxed);
        1
    } else {
        0
    }
}
