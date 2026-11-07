pub mod constants;
pub mod dataset;
pub mod batcher;
pub mod logger;

use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{
    SgdConfig,
    decay::WeightDecayConfig,
    lr_scheduler::module_lr_scheduler::ModuleLrSchedulerConfig,
    momentum::MomentumConfig,
};
use burn::prelude::Device;
use burn::train::metric::{AccuracyMetric, IterationSpeedMetric, LossMetric};
use burn::train::{Learner, SupervisedTraining};
use std::sync::Arc;
use model_core::model::TeacherCNN;
use crate::dataset::FashionMnistDataset;
use crate::batcher::FashionMnistBatcher;
use model_core::FashionMnistBatch;
use crate::constants::{BATCH_SIZE, EPOCH_NUMBER, LEARNING_RATE, SEED, WORKERS};
use crate::logger::ExperimentLogger;

fn evaluate(
    model: &TeacherCNN,
    dataloader: Arc<dyn DataLoader<FashionMnistBatch>>,
    device: &Device,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let mut total_loss = 0.0;
    let mut correct_predictions = 0;
    let mut total_samples = 0;
    let loss_fn = CrossEntropyLossConfig::new().init(device);

    for batch in dataloader.iter() {
        let batch = batch?;
        let images = batch.images.to_device(device);
        let targets = batch.targets.to_device(device);

        let logits = model.forward(images);
        let loss = loss_fn.forward(logits.clone(), targets.clone());

        let preds = logits.argmax(1).squeeze();
        let correct = preds
            .equal(targets.clone())
            .int()
            .sum()
            .into_scalar::<i64>() as usize;

        let batch_size = targets.dims()[0];
        total_loss += (loss.into_scalar::<f32>() as f64) * (batch_size as f64);
        correct_predictions += correct;
        total_samples += batch_size;
    }

    if total_samples == 0 {
        return Err("Validation data loader produced no samples".into());
    }

    let avg_loss = total_loss / (total_samples as f64);
    let accuracy = (correct_predictions as f64) / (total_samples as f64);

    if !accuracy.is_finite() || !avg_loss.is_finite() {
        return Err(format!(
            "Validation metrics are not finite: accuracy={accuracy}, loss={avg_loss}"
        )
        .into());
    }

    Ok((accuracy, avg_loss))
}

fn main() {
    let device = Device::default().autodiff();
    let artifact_dir = "./artifacts/fashion_mnist_guide";
    let exp_name = "teacher_cnn_sgd";

    let mut logger = ExperimentLogger::new(
        exp_name,
        artifact_dir,
        EPOCH_NUMBER,
        BATCH_SIZE,
        LEARNING_RATE,
        "SGD",
        WORKERS,
        SEED,
        format!("{:?}", device),
    );

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
        .batch_size(BATCH_SIZE)
        .shuffle(SEED)
        .num_workers(WORKERS)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(BATCH_SIZE)
        .num_workers(WORKERS)
        .build(valid_dataset);

    let model = TeacherCNN::init(&device);
    let optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_momentum(Some(MomentumConfig::new().with_momentum(0.963)))
        .init();
    let scheduler = ModuleLrSchedulerConfig::new(LEARNING_RATE.into())
        .init()
        .expect("Cannot create scheduler");

    let training = SupervisedTraining::new(
        artifact_dir,
        dataloader_train,
        dataloader_valid.clone(),
    )
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_valid_numeric(IterationSpeedMetric::new())
        .num_epochs(EPOCH_NUMBER);

    logger.start();

    let model_trained = training.launch(Learner::new(model, optimizer, scheduler));

    let model_path = format!("{}/model", artifact_dir);
    model_trained
        .model
        .clone()
        .save_file(&model_path)
        .expect("Cannot save model");

    let (valid_accuracy, valid_loss) =
        evaluate(&model_trained.model, dataloader_valid, &device)
            .expect("Cannot evaluate validation dataset");

    logger
        .complete_and_save(&model_path, valid_accuracy, valid_loss)
        .expect("Cannot save experiment JSON");
}