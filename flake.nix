{
  description = "REmbed - Calculate low dimensional weighted node embeddings, rewritten in rust";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    snn = {
      url = "github:wembed-pdf/snn";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, snn }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) snn.overlays.default ];
        toolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "clippy" "rust-analyzer"];
        };

        pkgs = import nixpkgs {
          inherit system overlays;
        };
      
        # Common R packages used across all shells
        rPackages = with pkgs.rPackages; [
          # Core data manipulation and visualization
          ggplot2
          dplyr
          janitor
          ggthemes
          scales
          ggforce
          svglite
          ggpattern
          
          # Database connectivity
          DBI
          RPostgres
          
          # Environment and utility packages
          dotenv
          readr
          
          # Color palettes and themes
          Polychrome
          viridis
          RColorBrewer
          
          # Additional useful packages for data analysis
          tidyr
          stringr
          lubridate
          # fast csv read
          data_table
          #fast ggsave
          ragg
          # Session
          jsonlite
          languageserver

          showtext

          patchwork
        ];
        
        # R with required packages
        R-with-packages = pkgs.rWrapper.override {
          packages = rPackages;
        };

        # Python with scikit-learn for sklearn feature
        python-with-sklearn = pkgs.python3.withPackages (ps: with ps; [
          numpy
          scikit-learn
          snnpy
          pandas
          h5py
        ]);

        # Common development tools
        devTools = with pkgs; [
          R-with-packages
          python-with-sklearn
          git
          wget
          
          lmodern
          gyre-fonts
        ];
        
        # C++ / native libraries the build links against. Kept in one list so the
        # devShell and the Docker image can never drift apart.
        nativeLibs = with pkgs; [
          boost.dev  # .dev output contains headers
          cgal       # CGAL development headers
          eigen      # Required by CGAL for Epick_d kernel
          gmp        # Required by CGAL
          mpfr       # Required by CGAL
        ];

        # Development tools (compilers, profilers, DB client, etc.). Shared too.
        buildTools = with pkgs; [
          # Development tools
          gdb
          valgrind
          gnuplot_qt
          clang
          pkgconf

          # Rust Development tools
          bacon
          samply
          cargo-show-asm
          toolchain
          sqlx-cli
          postgresql
        ];

        # The env vars that make the toolchain locate libclang, CGAL, and the
        # runtime shared libs. Defined once as an attrset so both the devShell
        # (via shellHook implicitly through mkShell) and the Docker image config
        # consume the identical values.
        buildEnv = {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.llvmPackages_latest.libclang.lib
            pkgs.stdenv.cc.cc.lib
            pkgs.gmp
            pkgs.mpfr
          ];
          LIBCLANG_PATH = pkgs.lib.makeLibraryPath [ pkgs.llvmPackages_latest.libclang.lib ];
          R_SHOWTEXT_FONTPATH = "${pkgs.lmodern}/share/fonts/opentype/public/lm";
          R_SHOWTEXT_HELVETICA_PATH = "${pkgs.gyre-fonts}/share/fonts/truetype";
        };

        # Env that the C++ feature builds (cgal / boost-rtree / wembed-snn /
        # nanoflann) need. In `nix develop` these are set implicitly by mkShell's
        # setup hooks (which route through Nix's cc-wrapper). The Docker image
        # invokes a bare `c++`/`clang` that does NOT read NIX_CFLAGS_COMPILE, so we
        # feed the include paths through channels the plain compiler + cc-rs honor.
        # Computed from the SAME nativeLibs list so it can't drift from the shell.
        #
        # Gotchas this handles, each verified against nixpkgs:
        #   - eigen ships eigen3.pc in share/pkgconfig, boost in lib/pkgconfig, and
        #     CGAL ships no .pc at all -> cover both dirs AND pass -isystem directly.
        #   - headers live in the `dev` output (lib.getDev), e.g. <eigen3/Eigen/Dense>.
        cppIncludeFlags = builtins.concatStringsSep " "
          (map (p: "-isystem ${pkgs.lib.getDev p}/include") nativeLibs);
        cppBuildEnv = {
          PKG_CONFIG_PATH = builtins.concatStringsSep ":" (
            (map (p: "${pkgs.lib.getDev p}/lib/pkgconfig") nativeLibs) ++
            (map (p: "${pkgs.lib.getDev p}/share/pkgconfig") nativeLibs)
          );
          # cc-rs appends CXXFLAGS/CFLAGS to its compiler command, and a bare
          # compiler reads them -> this is what actually gets the includes in.
          CXXFLAGS = cppIncludeFlags;
          CFLAGS = cppIncludeFlags;
          # bindgen's libclang reads this for header discovery.
          BINDGEN_EXTRA_CLANG_ARGS = cppIncludeFlags;
        };

        # Common shell hook
        commonShellHook = ''
          # Set R library path to use Nix packages
          export R_LIBS_USER=""
          export R_PROFILE_USER=".Rprofile"

          # Ensure plots directory exists
          mkdir -p plots
        '';
      in
      {
        devShells.default = pkgs.mkShell ({
          # C++ libraries go in buildInputs so Nix sets up include paths automatically
          buildInputs = devTools ++ nativeLibs;

          shellHook = commonShellHook;

          # Only development tools in packages
          packages = buildTools;

          NIX_ENFORCE_NO_NATIVE = false;
        } // buildEnv);

        packages.dockerImage = pkgs.dockerTools.buildLayeredImage {
          name = "rembed-env";
          tag = "latest";

          # Everything a researcher needs to build & run the experiments: the
          # native libs, the toolchain, R, Python, and a set of base-system
          # utilities that an otherwise-empty Nix image lacks (shell, coreutils,
          # CA certs so cargo can fetch crates, etc.).
          contents = devTools ++ nativeLibs ++ buildTools ++ (with pkgs; [
            bashInteractive
            coreutils
            gnugrep
            gnused
            gawk
            findutils
            which
            cacert
            pkgs.dockerTools.binSh   # provides /bin/sh
          ]);

          # A minimal dockerTools image has no /tmp, /work, or HOME. A build
          # environment needs all three: `cc`/lld write linker param files to
          # /tmp, cargo writes crate cache to HOME, and /work is the mount point.
          # These commands run in the image root at build time to seed them.
          extraCommands = ''
            mkdir -p tmp work root/.cargo
            chmod 1777 tmp
          '';

          config = {
            Cmd = [ "${pkgs.bashInteractive}/bin/bash" ];
            WorkingDir = "/work";
            Env = pkgs.lib.mapAttrsToList (name: value: "${name}=${value}") (buildEnv // cppBuildEnv // {
              # Baked-in equivalents of the shellHook exports.
              R_LIBS_USER = "";
              R_PROFILE_USER = ".Rprofile";
              # CA bundle so cargo/git over HTTPS work inside the container.
              SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              NIX_ENFORCE_NO_NATIVE = "false";
              # Scratch + HOME so compilers and cargo have writable working dirs.
              HOME = "/root";
              TMPDIR = "/tmp";
              CARGO_HOME = "/root/.cargo";
            });
          };
        };
      }
    );
}
