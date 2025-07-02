use rembed::{
    atree::ATree, dvec::DVec, embedder::EmbedderOptions, gpu_brute_force::GpuBruteForce,
    query::Embedder, *,
};

fn main() -> io::Result<()> {
    let graph_name = "rel8";
    // let graph_name = "bio-grid-fruitfly";

    let dim = 8;
    let dim_hint = 8;

    let graph_path = format!("data/{}/graph", graph_name);
    let graph = graph::Graph::parse_from_edge_list_file(&graph_path, dim, dim_hint)?;

    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use rembed::gpu_brute_force::gpu_nearest_neighbors;

    // Create sample data
    let positions = vec![
        DVec::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        DVec::new([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        DVec::new([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    ];

    let embedding = Embedding {
        positions,
        graph: &graph,
    };

    // Initialize GPU device
    let device = WgpuDevice::BestAvailable;

    // Run GPU neighbor search
    let neighbors = gpu_nearest_neighbors::<WgpuRuntime, 8>(
        &embedding, &device, 0,   // query index
        1.0, // radius
    );

    println!("Found {} GPU neighbors: {:?}", neighbors.len(), neighbors);

    let options = EmbedderOptions::default();
    let mut embedder: embedder::WEmbedder<GpuBruteForce<8>, 8> =
        embedder::WEmbedder::random(42, &graph, options);

    // takes wembed 03:04 for the first 100 iterations on rel8
    embedder.embed_with_callback(|i| {
        if i % 10 == 0 {
            println!("Iteration {i}")
        }
    });
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
