fn main() {
    // Tell cargo to tell rustc to link our C++ library
    println!("cargo:rustc-link-lib=static=wembed_snn_wrapper");
    println!("cargo:rustc-link-lib=stdc++");

    // Tell cargo to invalidate the built crate whenever source files change
    println!("cargo:rerun-if-changed=wembed_snn_wrapper.cpp");
    println!("cargo:rerun-if-changed=wembed_snn_wrapper.h");
    println!("cargo:rerun-if-changed=snn.cpp");
    println!("cargo:rerun-if-changed=snn.h");
    println!("cargo:rerun-if-changed=eign.cpp");
    println!("cargo:rerun-if-changed=eign.h");

    // Find Eigen3 using pkg-config
    let eigen3 = pkg_config::Config::new().probe("eigen3").ok();

    // Build the C++ wrapper and original SNN implementation
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("wembed_snn_wrapper.cpp")
        .file("snn.cpp")
        .file("eign.cpp")
        .include(".")  // For local headers (snn.h, eign.h)
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-march=native")
        .flag("-fPIC");

    // Add include paths from pkg-config if available
    if let Some(eigen3) = eigen3.as_ref() {
        for include in &eigen3.include_paths {
            build.include(include);
        }
    }

    build.compile("wembed_snn_wrapper");

    // Generate bindings with bindgen
    let mut bindgen_builder = bindgen::Builder::default()
        .header("wembed_snn_wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
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
        .allowlist_function("wembed_snn_.*")
        .allowlist_type("WembedSnnIndex")
        .allowlist_type("WembedSnnResult");

    // Add include paths to bindgen
    if let Some(eigen3) = pkg_config::Config::new().probe("eigen3").ok() {
        for include in &eigen3.include_paths {
            bindgen_builder = bindgen_builder.clang_arg(format!("-I{}", include.display()));
        }
    }

    let bindings = bindgen_builder
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("src/bindings.rs")
        .expect("Couldn't write bindings!");
}
