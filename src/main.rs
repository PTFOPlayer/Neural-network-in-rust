use matrix::Matrix;
use network::{Activation, Network};

pub mod error;
pub mod matrix;
pub mod network;

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

    println!("{:?}", network.forward_prop(Matrix::from(vec![0.0, 0.0])));
    println!("{:?}", network.forward_prop(Matrix::from(vec![1.0, 0.0])));
    println!("{:?}", network.forward_prop(Matrix::from(vec![0.0, 1.0])));
    println!("{:?}", network.forward_prop(Matrix::from(vec![1.0, 1.0])));
}
