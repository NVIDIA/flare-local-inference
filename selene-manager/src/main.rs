use selene_manager::{
    config::{self, Parsed},
    control,
};
use std::sync::{Arc, atomic::AtomicBool};

fn main() {
    if let Err(error) = start() {
        eprintln!("selene: {error}");
        std::process::exit(1);
    }
}

fn start() -> std::io::Result<()> {
    match config::parse_args(std::env::args().skip(1))? {
        Parsed::Help => {
            println!("{}", config::help_text());
            Ok(())
        }
        Parsed::Run(config) => {
            control::serve(config, Arc::new(AtomicBool::new(false)))
        }
    }
}
