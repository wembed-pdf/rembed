fn main() {
    // Tell cargo to tell rustc to link our C++ library
    println!("cargo:rustc-link-lib=static=boost_rtree_wrapper");
    println!("cargo:rustc-link-lib=stdc++");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=boost_rtree_wrapper.cpp");
    println!("cargo:rerun-if-changed=boost_rtree_wrapper.h");

    // Try to find Boost using pkg-config (works on NixOS and some Linux distributions)
    let boost = pkg_config::Config::new()
        .atleast_version("1.65")
        .probe("boost")
        .ok();

    // Build the C++ wrapper
    // The cc crate automatically picks up NIX_CFLAGS_COMPILE which contains include paths
    // for all packages in buildInputs (including boost)
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("boost_rtree_wrapper.cpp")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-march=native")
        .flag("-fPIC");

    // Add include paths from pkg-config if available
    if let Some(boost) = boost.as_ref() {
        for include in &boost.include_paths {
            build.include(include);
        }
    }

    build.compile("boost_rtree_wrapper");

    // Generate bindings with bindgen
    // Pass the same include paths to bindgen's clang
    let mut bindgen_builder = bindgen::Builder::default()
        .header("boost_rtree_wrapper.h")
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
        .allowlist_function("boost_rtree_.*")
        .allowlist_type("BoostRTreeIndex");

    // Add include paths to bindgen
    if let Some(boost) = boost {
        for include in &boost.include_paths {
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
