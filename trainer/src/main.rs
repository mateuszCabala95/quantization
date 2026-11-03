pub mod constants;
pub mod dataset;
pub mod batcher;

use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::optim::{AdamConfig, lr_scheduler::module_lr_scheduler::ModuleLrSchedulerConfig};
use burn::prelude::Device;
use burn::train::metric::{AccuracyMetric, IterationSpeedMetric, LossMetric, TopKAccuracyMetric};
use burn::train::{Learner, SupervisedTraining};

use model_core::model::TeacherCNN;
use crate::dataset::FashionMnistDataset;
use crate::batcher::FashionMnistBatcher;
use crate::constants::EPOCH_NUMBER;

fn main() {
    let device = Device::default().autodiff();
    let data_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data");

    let train_dataset = FashionMnistDataset::new(
        data_dir.join("train-images-idx3-ubyte"),
        data_dir.join("train-labels-idx1-ubyte"),
    )
        .expect("Cannot find training set.");

    let valid_dataset = FashionMnistDataset::new(
        data_dir.join("t10k-images-idx3-ubyte"),
        data_dir.join("t10k-labels-idx1-ubyte"),
    )
        .expect("Cannot find validation set.");

    let batcher_train = FashionMnistBatcher::new();
    let batcher_valid = FashionMnistBatcher::new();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(64)
        .shuffle(4021)
        .num_workers(4)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(64)
        .num_workers(4)
        .build(valid_dataset);

    let model = TeacherCNN::init(&device);
    let optimizer = AdamConfig::new().init();
    let scheduler = ModuleLrSchedulerConfig::new(1e-4.into())
        .init()
        .expect("Cannot create scheduler");

    let training = SupervisedTraining::new(
        "./artifacts/fashion_mnist_guide",
        dataloader_train,
        dataloader_valid,
    )
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_valid_numeric(IterationSpeedMetric::new())
        .num_epochs(EPOCH_NUMBER);

    let model_trained = training.launch(Learner::new(model, optimizer, scheduler));

    model_trained
        .model
        .save_file("./artifacts/fashion_mnist_guide/model")
        .expect("Cannot save model");
}