use benchmark::GraphGenerator;

fn main() {
    let graph_generator = GraphGenerator {
        girgs_path: "../../girgs/build/genhrg".to_string(),
        output_path: "../data/generated".to_string(),
    };
    graph_generator.generate();
}
