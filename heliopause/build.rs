fn main() {
    println!("cargo:rustc-check-cfg=cfg(openssl3)");
    if std::env::var("DEP_OPENSSL_VERSION_NUMBER")
        .ok()
        .and_then(|value| u64::from_str_radix(&value, 16).ok())
        .is_some_and(|version| version >= 0x3000_0000)
    {
        println!("cargo:rustc-cfg=openssl3");
    }
}
