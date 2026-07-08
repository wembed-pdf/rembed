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

          cowplot
          patchwork

          tikzDevice
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
        ]);

        # Common development tools
        devTools = with pkgs; [
          R-with-packages
          python-with-sklearn
          git
          
          lmodern
          gyre-fonts
        ];
        
        # Common shell hook
        commonShellHook = ''
          # Set R library path to use Nix packages
          export R_LIBS_USER=""
          export R_PROFILE_USER=".Rprofile"
          
          # Ensure plots directory exists
          mkdir -p plots

          # Set R library path to use Nix packages
          export R_LIBS_USER=""
          export R_PROFILE_USER=".Rprofile"

          # Create a font path for showtext to use the lmodern fonts from Nix
          export R_SHOWTEXT_FONTPATH="${pkgs.lmodern}/share/fonts/opentype/public/lm"
          export R_SHOWTEXT_HELVETICA_PATH="${pkgs.gyre-fonts}/share/fonts/truetype"
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          # C++ libraries go in buildInputs so Nix sets up include paths automatically
          buildInputs = devTools ++ (with pkgs; [
            boost.dev  # .dev output contains headers
            cgal       # CGAL development headers
            eigen      # Required by CGAL for Epick_d kernel
            gmp        # Required by CGAL
            mpfr       # Required by CGAL
          ]);

          shellHook = commonShellHook;

          # Only development tools in packages
          packages = with pkgs; [
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

          NIX_ENFORCE_NO_NATIVE=false;

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.llvmPackages_latest.libclang.lib
            pkgs.stdenv.cc.cc.lib
            pkgs.gmp
            pkgs.mpfr
          ];
          LIBCLANG_PATH = pkgs.lib.makeLibraryPath [ pkgs.llvmPackages_latest.libclang.lib ];
        };
      }
    );
}
