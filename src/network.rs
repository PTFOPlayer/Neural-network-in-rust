use std::{
    fs::{self, File},
    io::{self, Write},
};

use serde::{Deserialize, Serialize};

use crate::{activation::Activation, error::NetworkError, matrix::Matrix};

#[derive(Serialize, Deserialize)]
pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    #[serde(skip)]
    activation: Activation,
    rate: f32,
}

impl Network {
    pub fn new(layers: Vec<usize>, activation: Activation, rate: f32) -> Network {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random_pregen(layers[i + 1], layers[i]));
            biases.push(Matrix::random_pregen(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: Vec::new(),
            activation,
            rate,
        }
    }

    pub fn forward_prop(&mut self, input: Matrix) -> Result<Matrix, NetworkError> {
        let mut current = input;

        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .dot_prod(&current)
                .map_err(NetworkError::map_fwd)?
                .add(&self.biases[i])?
                .map_fn(self.activation.fx);

            self.data.push(current.clone());
        }

        Ok(current)
    }

    #[must_use]
    pub fn back_prop(&mut self, input: Matrix, target: Matrix) -> Result<(), NetworkError> {
        let mut error = target.sub(&input)?;
        let mut grad = input.clone().map_fn(self.activation.fdx);

        let rate = self.rate;
        for i in (0..self.layers.len() - 1).rev() {
            grad = grad
                .element_mul(&error)
                .map_err(NetworkError::map_bck)?
                .map_fn(|x| x * rate);

            self.weights[i] = self.weights[i].add(
                &grad
                    .dot_prod(&self.data[i].traspose())
                    .map_err(NetworkError::map_bck)?,
            )?;
            self.biases[i] = self.biases[i].add(&grad)?;

            error = self.weights[i]
                .traspose()
                .dot_prod(&error)
                .map_err(NetworkError::map_bck)?;

            grad = self.data[i].map_fn(self.activation.fdx);
        }

        Ok(())
    }

    pub fn train(
        mut self,
        input: Vec<Vec<f32>>,
        target: Vec<Vec<f32>>,
        epoch: usize,
    ) -> Result<Network, NetworkError> {
        for _ in 1..=epoch {
            for j in 0..input.len() {
                let out = self.forward_prop(Matrix::from(input[j].clone()))?;
                self.back_prop(out, Matrix::from(target[j].clone()))?;
            }
        }

        Ok(self)
    }

    pub fn save_to_file(&self, path: &str) -> Result<(), io::Error> {
        let data = serde_json::ser::to_string(self)?;
        let mut file = File::create(path)?;
        file.write(data.as_bytes())?;
        file.flush()?;

        Ok(())
    }

    pub fn read_from_file(path: &str) -> Result<Self, io::Error> {
        Ok(serde_json::de::from_slice(&fs::read(path)?)?)
    }
}
