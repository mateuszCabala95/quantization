use burn::prelude::*;

pub struct FashionMnistBatch {
    pub images: Tensor<4>,
    pub targets: Tensor<1, Int>,
}
