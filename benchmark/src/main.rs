use benchmark::GraphGenerator;
use benchmark::PositionGenerator;

fn main() {
    let graph_generator = GraphGenerator {
        girgs_path: "../../girgs/build/genhrg".to_string(),
        output_path: "../data/generated".to_string(),
    };
    graph_generator.generate();

    let position_generator = PositionGenerator {
        wembed_path: "./../../wembed/release/bin/cli_wembed".to_string(),
        graphs_path: "../data/generated/wembed_graphs".to_string(),
        output_path: "../data/generated/positions".to_string(),
    };
    position_generator.change_graphs("../data/generated/graphs".to_string());
    position_generator.generate();
}
