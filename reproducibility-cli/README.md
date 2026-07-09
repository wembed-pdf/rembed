Reproducibility CLI
-----------

This is a command line interface (CLI) to help reproduce figures and tables from the paper "Benchmarking and Engineering Data Structures for Spherical Range Queries"

How to Use
-----------

0. Check whether the figure/table you want to reproduce is [supported by this CLI](#status-of-reproducibility).
1. Clone this repository and navigate to the `reproducibility-cli` directory.

   ```git clone https://github.com/wembed-pdf/rembed```

2. This project uses a nix flake to create a reproducible environment. 

   To use the flake, you need to have [nix](https://nixos.org/download.html) installed. Then, you can run the following command to enter the environment:

   ```nix develop```

   Alternatively, you can use the prebuilt Docker image to get the same environment without installing nix. To do so, you need to have [docker](https://docs.docker.com/get-docker/) installed. Then, run the following command from the repository root to start a shell inside the environment (the current directory is mounted at `/work`):

   ```docker run -it --cap-add PERFMON -v "$PWD:/work" docker.io/truedoctor/rembed-env:latest```

   The `--cap-add PERFMON` flag is required: the benchmark measures CPU cycles and instructions via Linux perf counters (`perf_event_open`), which containers block by default. If you still see `Failed to create perf event group: PermissionDenied`, your setup needs one or both of the following:
   - Add `--security-opt seccomp=unconfined` to the `docker run` command, since Docker's default seccomp profile also filters the `perf_event_open` syscall.
   - On the host, lower the perf paranoia level: `sudo sysctl kernel.perf_event_paranoid=1`.

3. Run the CLI for the figure/table you want to reproduce. For example, to reproduce Figure 3, run:

   ```cargo run --release reproduce --figure 4 --structures sprk kiddo```

   The cli will automatically download the required datasets and run the benchmarks. 
   If you prefer to download the datasets manually, you can do so by following the instructions in the [datasets](#datasets) section.

Useful Tips / Troubleshooting
-----------

- The default settings for the CLI are set to reproduce the figures/tables thoroughly, which may take a long time. If you want to reproduce the figures/tables faster and with less precision, you can use the `--fast` flag. Alternatively, you can use the `--node-counts`, `--dimensions`, and `--embedding-seeds` flags to only reproduce parts of the figures/tables. For example, to reproduce Figure 3 with only 100,000 nodes and 2 or 8 dimensions and a single embedding seed, run:

   ```cargo run --release reproduce --figure 4 --structures sprk kiddo --node-counts 100000 --dimensions 2 8 --embedding-seeds 12```

- To use python-based backends run 
```sh
   cargo run --release --features python reproduce --figure 4 --structures sklearn_kdtree py_snn
```



Status of Reproducibility
-----------

| Figure/Table | Reproducibility-CLI | Benchmark-CLI | Comment
|---------|:------:|:------:|:------:|
| Figure 1 |  |  | Not a benchmark |
| Figure 2 |  |  | Not a benchmark |
| Figure 3 | ✨ | ✅ |  |
| Figure 4 | ✨ | ✅ |  |
| Figure 5 |  |  | Simulation Crate |
| Figure 6 |  |  | SPRK Crate |
| Figure 7 | ✨ | ✅ |  |
| Figure 8 |  |  | Embedder-CLI |
| Figure 9 | ✨ | ✅ |  |
| Figure 10 |  | ✅ |  |
| Figure 11 | ✨ | ✅ |  |
| Table 1 |  |  | Not a benchmark |
| Table 2 | ✨ | ✅ |  |
| Table 3 | ✨ | ✅ |  |
| Table 4 | ✨ | ✅ |  |
| Table 5 | ✨ | ✅ |  |
| Table 6 | ✨ | ✅ |  |
| Table 7 | ✨ | ✅ |  |


Datasets
-----------

Create a data directory and download the required datasets. The data directory should have the following structure:

```
reproducibility-cli
├── data
      ├── embedding(https://zenodo.org/records/21243483/files/embedding_data.zip?download=1)
      │   ├── embedding_data
      │   ├── embedding_metadata.csv
      ├── distributions(https://zenodo.org/records/21243483/files/distributions_data.zip?download=1)
      │   ├── distributions_data
      ├── realworld
         ├── clustering_data
         ├── nearest_neighbor_data 
         ├── poi_data (https://zenodo.org/records/21243483/files/poi_data.zip?download=1)
```

- The clustering data stem from the [openML datasets](https://www.openml.org).
- The nearest neighbor data stem from the [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks).

There are download helper scripts in the reproducibility-cli/data/download_scripts directory. You can run the scripts to download the datasets automatically. These are also the scripts that are run by the CLI when you run the `reproduce` command. 


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
