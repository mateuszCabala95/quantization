use burn::prelude::*;
use burn::nn::{
    conv::{Conv2d},
    pool::{MaxPool2d},
    Dropout, Linear, Relu,
};

#[derive(Config, Debug)]
pub struct TeacherCNNConfig {
    #[config(default = 10)]
    pub num_classes: usize,
    #[config(default = 0.5)]
    pub dropout_rate: f64,
}

#[derive(Module, Debug)]
pub struct TeacherCNN {
    conv1: Conv2d,
    pool1: MaxPool2d,

    conv2: Conv2d,
    pool2: MaxPool2d,

    linear1: Linear,
    dropout: Dropout,
    linear2: Linear,

    activation: Relu,
}

impl TeacherCNN {
    pub fn init(device: &Device) -> Self {
        todo!()
    }

    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        todo!()
    }
}