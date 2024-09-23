pub struct TrainingData {
    pub inputs: Vec<f64>,
    pub target: Vec<f64>,
}

impl TrainingData {
    pub fn new(inputs: Vec<f64>, target: Vec<f64>) -> Self {
        Self { inputs, target }
    }
}
