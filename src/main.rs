use activation::Activation;
use matrix::Matrix;
use network::Network;

pub mod error;
pub mod matrix;
pub mod network;
pub mod activation;

fn main() {
    // test for XOR
    let input = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let target = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let mut network = Network::new(vec![2, 3, 1], Activation::default(), 0.5);

    network = network.train(input, target, 200000).unwrap();

    network.save_to_file("./data.json").unwrap();

    let mut network = Network::read_from_file("./data.json").unwrap();    

    println!("{:?}", network.forward_prop(Matrix::from(vec![0.0, 0.0])));
    println!("{:?}", network.forward_prop(Matrix::from(vec![1.0, 0.0])));
    println!("{:?}", network.forward_prop(Matrix::from(vec![0.0, 1.0])));
    println!("{:?}", network.forward_prop(Matrix::from(vec![1.0, 1.0])));
}
