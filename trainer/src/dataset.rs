use burn::data::dataset::{Dataset, DatasetError};

pub struct FashionMnistItem{
    pub image: [u8; 784],
    pub label: u8
}

pub struct FashionMnistDataset{
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
    pub length: usize
}

impl Dataset<FashionMnistItem> for FashionMnistDataset{
    fn get(&self, index: usize) -> Result<FashionMnistItem, DatasetError> {
       if index >= self.length {
           return Err(DatasetError::new(std::io::Error::new(
               std::io::ErrorKind::InvalidInput,
               "Index is out of range"
           )));
       }

        let img_start = index * 784;
        let img_end = img_start + 784;

        let mut image_array = [0u8; 784];
        image_array.copy_from_slice(&self.images[img_start..img_end]);

        Ok(FashionMnistItem{
            image: image_array,
            label: self.labels[index]
        })
    }

    fn len(&self) -> usize {
        self.length
    }
}

impl FashionMnistDataset {
    pub fn new(images_path: &str, labels_path: &str) -> std::io::Result<Self> {
        let images_raw = std::fs::read(images_path)?;
        let labels_raw = std::fs::read(labels_path)?;

        let images = images_raw[16..].to_vec();
        let labels = labels_raw[8..].to_vec();

        let length = labels.len();

        if images.len() != length * 784 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid file size",
            ));
        }

        Ok(Self {
            images,
            labels,
            length,
        })
    }
}
