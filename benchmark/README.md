Benchmarkk CLI
-----------

This is a command line interface (CLI) to benchmark spatial index implementations for spherical range queries. 

How to Use
-----------

0. This project uses a nix flake to create a reproducible environment. 

   To use the flake, you need to have [nix](https://nixos.org/download.html) installed. Then, you can run the following command to enter the environment:

   ```nix develop```

   Alternatively, you can install the dependencies manually, take a look at the `flake.nix` file to see which dependencies are required. Most dependencies are for competitor implementations and are not strictly necessary to get the CLI running. 

1. Run the CLI run:

   ```cargo run --release benchmark```

   There are several subcommands available

Useful Tips
-----------

- To use python-based backends run 
```sh
   cargo run --release --features python benchmark
```

Bug Reports
-----------

We encourage you to report any problems with the reproducibility-cli via the [github issue tracking system](https://github.com/wembed-pdf/rembed/issues). 
For issues regarding the SPRK crate, please use the [sprk github issue tracking system](https://github.com/wembed-pdf/sprk)

License
-----------

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
For license information regarding the datasets, please refer to the respective dataset sources.

If you use Rembed in your research, please cite the following paper:

```
@misc{bläsius2026benchmarkingengineeringdatastructures,
      title={Benchmarking and Engineering Data Structures for Spherical Range Queries}, 
      author={Thomas Bläsius and Jean-Pierre von der Heydt and Tobias Kempf and Dennis Kobert and Nikolai Maas},
      year={2026},
      eprint={2607.07367},
      archivePrefix={arXiv},
      primaryClass={cs.CG},
      url={https://arxiv.org/abs/2607.07367}, 
}
```
