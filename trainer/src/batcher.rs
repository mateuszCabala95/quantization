use burn::backend::Backend;
use burn::prelude::*;
use burn::data::dataloader::batcher::Batcher;
use crate::dataset::FashionMnistItem;

pub struct FashionMnistBatch{
    pub images: Tensor<4>,
    pub targets: Tensor<1, Int>,
}

pub struct FashionMnistBatcher{
    device: Device,
}

impl FashionMnistBatcher{
    pub fn new(device: Device) -> Self{
        Self {device}
    }
}

impl Batcher<FashionMnistItem, FashionMnistBatch> for FashionMnistBatcher{
    fn batch(&self, items: Vec<FashionMnistItem>, device: &Device) -> FashionMnistBatch {
        let batch_size = items.len();

        let mut images_vec = Vec::with_capacity(batch_size * 784);
        let mut targets_vec = Vec::with_capacity(batch_size);

        for item in items.iter() {
            images_vec.extend(item.image.iter().map(|&b| b as f32));
            targets_vec.push(item.label as i64);
        }

        let images_data = TensorData::new(images_vec, [batch_size * 784]);
        let targets_data = TensorData::new(targets_vec, [batch_size]);

        let images = Tensor::<1>::from_data(images_data, &self.device).reshape([batch_size, 1, 28, 28]) / 255.0;

        let targets = Tensor::<1, Int>::from_data(targets_data, &self.device);

        FashionMnistBatch{
            images,
            targets
        }
    }
}


