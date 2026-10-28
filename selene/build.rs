fn main() {
    let proto = "proto/selene.proto";
    println!("cargo:rerun-if-changed={proto}");
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&[proto], &["proto"])
        .expect("compile the selene pool protocol");
}
