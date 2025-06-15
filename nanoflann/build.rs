fn main() {
    // Tell cargo to tell rustc to link our C++ library
    println!("cargo:rustc-link-lib=static=nanoflann_wrapper");
    println!("cargo:rustc-link-lib=stdc++");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=nanoflann_wrapper.cpp");
    println!("cargo:rerun-if-changed=nanoflann_wrapper.h");
    println!("cargo:rerun-if-changed=nanoflann.hpp");

    // Compile the C++ wrapper
    cc::Build::new()
        .cpp(true)
        .file("nanoflann_wrapper.cpp")
        .include(".") // Include current directory for nanoflann.hpp
        .flag("-std=c++14")
        .flag("-O3")
        .flag("-march=native")
        .flag("-fPIC")
        .compile("nanoflann_wrapper");

    // Generate bindings with bindgen
    let bindings = bindgen::Builder::default()
        .header("nanoflann_wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Tell bindgen to generate Rust-friendly names
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
        // .derive_clone(true)
        // Don't generate bindings for standard library types
        .blocklist_type("FILE")
        .blocklist_type("_IO_.*")
        .blocklist_function("strtold")
        .blocklist_function("qecvt")
        .blocklist_function("qfcvt")
        .blocklist_function("qgcvt")
        .blocklist_function("qecvt_r")
        .blocklist_function("qfcvt_r")
        // Generate only what we need
        .allowlist_function("nanoflann_.*")
        .allowlist_type("NanoflannIndex")
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("src/bindings.rs")
        .expect("Couldn't write bindings!");
}
