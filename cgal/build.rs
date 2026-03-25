fn main() {
    // Tell cargo to tell rustc to link our C++ library and CGAL dependencies
    println!("cargo:rustc-link-lib=static=cgal_wrapper");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=gmp"); // CGAL dependency
    println!("cargo:rustc-link-lib=mpfr"); // CGAL dependency

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=cgal_wrapper.cpp");
    println!("cargo:rerun-if-changed=cgal_wrapper.h");

    // Find dependencies using pkg-config (works on NixOS and Linux distributions)
    let cgal = pkg_config::Config::new()
        .atleast_version("5.0")
        .probe("CGAL")
        .ok();

    // CGAL's Epick_d kernel requires Eigen3
    let eigen3 = pkg_config::Config::new().probe("eigen3").ok();

    // Build the C++ wrapper
    // The cc crate automatically picks up NIX_CFLAGS_COMPILE which contains include paths
    // for all packages in buildInputs (cgal, eigen, gmp, mpfr)
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("cgal_wrapper.cpp")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-march=native")
        .flag("-fPIC")
        .flag("-frounding-math"); // Required by CGAL

    // Add include paths from pkg-config if available
    if let Some(cgal) = cgal.as_ref() {
        for include in &cgal.include_paths {
            build.include(include);
        }
    }

    if let Some(eigen3) = eigen3.as_ref() {
        for include in &eigen3.include_paths {
            build.include(include);
        }
    }

    build.compile("cgal_wrapper");

    // Generate bindings with bindgen
    // Pass the same include paths to bindgen's clang
    let mut bindgen_builder = bindgen::Builder::default()
        .header("cgal_wrapper.h")
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
        .allowlist_function("cgal_kdtree_.*")
        .allowlist_type("CgalKdTreeIndex")
        .allowlist_type("CgalKdTreeResult");

    // Add include paths to bindgen
    if let Ok(cgal) = pkg_config::Config::new().probe("CGAL") {
        for include in &cgal.include_paths {
            bindgen_builder = bindgen_builder.clang_arg(format!("-I{}", include.display()));
        }
    }

    if let Ok(eigen3) = pkg_config::Config::new().probe("eigen3") {
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
