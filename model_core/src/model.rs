use burn::prelude::*;
use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    pool::{MaxPool2d, MaxPool2dConfig},
    Dropout, Linear, Relu, LinearConfig, DropoutConfig
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
       TeacherCNN{
           conv1: Conv2dConfig::new([1, 64], [2,2]).init(device),
           pool1: MaxPool2dConfig::new([2,2]).init(),

           conv2: Conv2dConfig::new([64, 128], [2,2]).init(device),
           pool2: MaxPool2dConfig::new([2,2]).init(),

           linear1: LinearConfig::new(6*6*128, 512).init(device),
           dropout: DropoutConfig::new(0.5).init(),
           linear2: LinearConfig::new(512,10).init(device),

           activation: Relu::new()
       }
    }

    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {

        // Layer 1
        let channel = self.conv1.forward(images);
        let channel = self.activation.forward(channel);
        let channel = self.pool1.forward(channel);

        //Layer 2
        let channel = self.conv2.forward(channel);
        let channel = self.activation.forward(channel);
        let channel = self.pool2.forward(channel);

        let channel = channel.flatten::<2>(1,3);

        //Layer 3
        let channel = self.linear1.forward(channel);
        let channel = self.activation.forward(channel);
        let channel = self.dropout.forward(channel);

        self.linear2.forward(channel)
    }
}