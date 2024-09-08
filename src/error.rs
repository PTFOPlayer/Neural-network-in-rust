#[derive(Debug, Clone)]
pub enum NetworkError {
    AddError,
    DotError,
    EMulError,
    SubError,
    BackProp(Box<NetworkError>),
    ForwardProp(Box<NetworkError>),
    BackTracked(String, Box<NetworkError>)
}

impl NetworkError {
    pub fn map_fwd(err: NetworkError) -> NetworkError {
        NetworkError::ForwardProp(Box::new(err))
    }
    pub fn map_bck(err: NetworkError) -> NetworkError {
        NetworkError::ForwardProp(Box::new(err))
    }
}