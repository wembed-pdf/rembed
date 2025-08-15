use rembed::{
    atree::ATree, dim_reduction::LayeredLsh, embedder::EmbedderOptions, query::Embedder, *,
};

fn main() -> io::Result<()> {
    let graph_name = "rel8";
    // let graph_name = "bio-grid-fruitfly";

    let dim = 8;
    let dim_hint = 8;

    // let graph_path = format!("data/{}/graph", graph_name);
    let graph_path = format!("data/{}/graph", graph_name);
    // let graph = "data/generated/graphs/1084_girg_n-1000_deg-25_dim-2_ple-2.5_alpha-inf_wseed-14_pseed-132_sseed-1402";
    let graph = "data/generated/graphs/19_girg_n-1000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    let graph = graph::Graph::parse_from_edge_list_file(&graph, dim, dim_hint)?;

    let options = EmbedderOptions::default();
    let mut embedder: embedder::WEmbedder<LayeredLsh<8>, 8> =
        embedder::WEmbedder::random(42, &graph, options);

    // takes wembed 03:04 for the first 100 iterations on rel8
    // for i in 0..1000 {
    // embedder.calculate_step();
    embedder.embed_with_callback(|i| {
        if i % 10 == 0 && i > 0 {
            println!("Iteration {i}");
            // let pos = embedder.positions();
            // let embedding = Embedding {
            //     positions: pos.to_vec(),
            //     graph: &graph,
            // };
            // let (percision, recall) = embedder.spatial_index.graph_statistics();
            // let f1 = 2. / (recall.recip() + percision.recip());
            // println!("i: , percision: {percision}, recall: {recall}, f1: {f1}");
        }
    });
    // }
    let pos = embedder.positions();
    let embedding = Embedding {
        positions: pos.to_vec(),
        graph: &graph,
    };
    let (percision, recall) = embedding.graph_statistics();
    let f1 = 2. / (recall.recip() + percision.recip());
    println!("i: , percision: {percision}, recall: {recall}, f1: {f1}");

    Ok(())
}
