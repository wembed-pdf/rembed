{
  description = "ANN dataset downloader + converter environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        python = pkgs.python3.withPackages (ps: with ps; [
          numpy
          pandas
          h5py
          tqdm
        ]);

      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pkgs.wget
            pkgs.gcc   # sometimes needed for h5py runtime linking
          ];

          shellHook = ''
            echo "✅ Python environment ready"
            echo "Run your script with: python script.py"
          '';
        };

        packages.default = pkgs.stdenv.mkDerivation {
          name = "ann-dataset-script";

          src = ./.;

          buildInputs = [
            python
            pkgs.wget
          ];

        };

      }
    );
}
