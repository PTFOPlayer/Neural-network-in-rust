use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::error::NetworkError;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn random_pregen(rows: usize, cols: usize) -> Matrix {
        let mut data = Vec::with_capacity(rows * cols);

        let mut rng = rand::thread_rng();
        for _ in 0..rows * cols {
            data.push(rng.gen_range(0.0..1.0));
        }

        Matrix { rows, cols, data }
    }

    pub fn add(&self, rhs: &Matrix) -> Result<Matrix, NetworkError> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(NetworkError::AddError);
        }
        let size = self.rows * self.cols;

        let mut data = Vec::with_capacity(size);

        for i in 0..size {
            data.push(self.data[i] + rhs.data[i]);
        }

        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }
    pub fn sub(&self, rhs: &Matrix) -> Result<Matrix, NetworkError> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(NetworkError::SubError);
        }
        let size = self.rows * self.cols;

        let mut data = Vec::with_capacity(size);

        for i in 0..size {
            data.push(self.data[i] - rhs.data[i]);
        }

        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    pub fn dot_prod(&self, rhs: &Matrix) -> Result<Matrix, NetworkError> {
        if self.cols != rhs.rows {
            return Err(NetworkError::DotError);
        }

        let mut data = vec![0.0; self.rows * rhs.cols];

        for i in 0..self.rows {
            for j in 0..rhs.cols {
                data[i * rhs.cols + j] = (0..self.cols).map(|k|{
                    self.data[i * self.cols + k] * rhs.data[k * rhs.cols + j]
                }).sum();
            }
        }

        Ok(Matrix {
            rows: self.rows,
            cols: rhs.cols,
            data,
        })
    }

    pub fn element_mul(&self, rhs: &Matrix) -> Result<Matrix, NetworkError> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(NetworkError::EMulError);
        }
        
        let size = self.rows * self.cols;

        let mut data = Vec::with_capacity(size);

        for i in 0..size {
            data.push(self.data[i] * rhs.data[i]);
        }

        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    pub fn traspose(&self) -> Matrix {
        let mut data = vec![0.0; self.cols * self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }

        Matrix {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    pub fn map_fn(&mut self, func: impl Fn(f64) -> f64) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| func(*x)).collect(),
        }
    }
}

impl From<Vec<f64>> for Matrix {
    fn from(data: Vec<f64>) -> Self {
        Matrix {
            rows: data.len(),
            cols: 1,
            data,
        }
    }
}
