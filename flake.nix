{
  description = "REmbed - Calculate low dimensional weighted node embeddings, rewritten in rust";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        toolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "clippy" "rust-analyzer"];
        };

        pkgs = import nixpkgs {
          inherit system overlays;
        };
      
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # Development tools
            gdb
            valgrind
            gnuplot
            clang
            pkgconf

            # Rust Development tools
            bacon
            samply
            toolchain
            sqlx-cli
          ];


          NIX_ENFORCE_NO_NATIVE=false;

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.llvmPackages_latest.libclang.lib pkgs.stdenv.cc.cc.lib ];
          LIBCLANG_PATH = pkgs.lib.makeLibraryPath [ pkgs.llvmPackages_latest.libclang.lib ];
        };
      }
    );
}
