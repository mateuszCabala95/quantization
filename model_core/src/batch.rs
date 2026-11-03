use burn::prelude::*;

#[derive(Clone, Debug)]
pub struct FashionMnistBatch {
    pub images: Tensor<4>,
    pub targets: Tensor<1, Int>,
}
