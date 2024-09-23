use activation::Activation;
use matrix::Matrix;
use network::Network;
use training_data::TrainingData;

pub mod activation;
pub mod error;
pub mod matrix;
pub mod network;
pub mod training_data;

fn main() {
    // test for XOR
    let training_data = vec![
        TrainingData::new(vec![0.0, 0.0], vec![0.0]),
        TrainingData::new(vec![0.0, 1.0], vec![1.0]),
        TrainingData::new(vec![1.0, 0.0], vec![1.0]),
        TrainingData::new(vec![1.0, 1.0], vec![0.0]),
    ];

    let mut network = Network::new(vec![2, 4, 1], Activation::default(), 0.5);

    network = network.train(training_data, 200000).unwrap();

    network.save_to_file("./data.json").unwrap();

    let mut network = Network::read_from_file("./data.json").unwrap();

    println!("{:?}", network.forward_prop(Matrix::from(vec![0.0, 0.0])));
    println!("{:?}", network.forward_prop(Matrix::from(vec![1.0, 0.0])));
    println!("{:?}", network.forward_prop(Matrix::from(vec![0.0, 1.0])));
    println!("{:?}", network.forward_prop(Matrix::from(vec![1.0, 1.0])));
}
