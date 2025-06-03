use benchmark::GraphGenerator;
use benchmark::PositionGenerator;

use dotenv::dotenv;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let collector = tracing_subscriber::fmt()
        // filter spans/events with level INFO or higher.
        .with_max_level(tracing::Level::INFO)
        // build but do not install the subscriber.
        .finish();
    // install global collector configured based on RUST_LOG env var.
    tracing::subscriber::set_global_default(collector).unwrap();

    let graph_generator = GraphGenerator {
        girgs_path: env::var("GIRGS_PATH").unwrap_or("../../girgs/build/genhrg".to_string()),
        output_path: "../data/generated/graphs".to_string(),
    };
    graph_generator.generate().await?;

    let position_generator = PositionGenerator {
        wembed_path: "./../../wembed/release/bin/cli_wembed".to_string(),
        graphs_path: "../data/generated/wembed_graphs".to_string(),
        output_path: "../data/generated/positions".to_string(),
    };
    position_generator.change_graphs("../data/generated/graphs".to_string());
    position_generator.generate();
    Ok(())
}
