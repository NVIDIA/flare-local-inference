fn main() {
    #[cfg(feature = "runtime-package")]
    {
        println!("cargo:rerun-if-changed=proto/runtime_package.proto");
        let mut config = prost_build::Config::new();
        config
            .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
            .file_descriptor_set_path(
                std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
                    .join("runtime_package_descriptor.bin"),
            )
            .field_attribute(".flare.runtime.ExecutionGraph.packed_token_capacity", "#[serde(default)]")
            .compile_protos(&["proto/runtime_package.proto"], &["proto"])
            .expect("compile runtime package protobuf schema");
    }
    #[cfg(feature = "selene")]
    {
        println!("cargo:rerun-if-changed=proto/selene.proto");
        tonic_prost_build::configure()
            .build_client(true)
            .build_server(true) // server stubs back the #[cfg(test)] fake
            .server_mod_attribute(".", "#[cfg(test)]")
            .compile_protos(&["proto/selene.proto"], &["proto"])
            .expect("compile selene protobuf schema");
    }
    #[cfg(feature = "roce")]
    {
        println!("cargo:rerun-if-changed=native/roce_proxy.c");
        cc::Build::new()
            .file("native/roce_proxy.c")
            .std("gnu11")
            .compile("flare_roce_proxy");
        println!("cargo:rustc-link-lib=ibverbs");
        println!("cargo:rustc-link-lib=pthread");
    }
}
