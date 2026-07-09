Reproducibility CLI
-----------

This is a command line interface (CLI) to help reproduce figures and tables from the paper "Benchmarking and Engineering Data Structures for Spherical Range Queries"

How to Use
-----------

0. Check whether the figure/table you want to reproduce is [supported by this CLI](#status-of-reproducibility).
1. Clone this repository and navigate to the `reproducibility-cli` directory.

   ```git clone https://github.com/wembed-pdf/rembed```

2. Create a data directory and download the required datasets. The data directory should have the following structure:

   ```
   reproducibility-cli
   ├── data
       ├── embedding(https://zenodo.org/records/21243483)
       │   ├── embedding_data
       │   ├── embedding_metadata.csv
       ├── distributions(https://zenodo.org/records/21243483)
       │   ├── distribution_data
       ├── realworld(https://zenodo.org/records/21243483)
           ├── clustering_data
           ├── nearest_neighbor_data
           ├── poi_data
   ```

3. This project uses a nix flake to create a reproducible environment. 

   To use the flake, you need to have [nix](https://nixos.org/download.html) installed. Then, you can run the following command to enter the environment:

   ```nix develop```

   Alternatively, you can install the dependencies manually, take a look at the `flake.nix` file to see which dependencies are required. Most dependencies are for competitor implementations and are not strictly necessary to get the CLI running. 

4. Run the CLI for the figure/table you want to reproduce. For example, to reproduce Figure 3, run:

   ```cargo run --release reproduce --figure 4 --structures sprk kiddo```


Useful Tips
-----------

- The default settings for the CLI are set to reproduce the figures/tables thoroughly, which may take a long time. If you want to reproduce the figures/tables faster and with less precision, you can use the `--fast` flag. Alternatively, you can use the `--node-counts`, `--dimensions`, and `--embedding-seeds` flags to only reproduce parts of the figures/tables. For example, to reproduce Figure 3 with only 100,000 nodes and 2 or 8 dimensions and a single embedding seed, run:

   ```cargo run --release reproduce --figure 4 --structures sprk kiddo --node-counts 100000 --dimensions 2 8 --embedding-seeds 12```



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