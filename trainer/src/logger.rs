use std::fs;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use serde::Serialize;

#[derive(Serialize)]
pub struct Hyperparameters {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub optimizer: String,
    pub workers: usize,
    pub seed: u64,
}

#[derive(Serialize)]
pub struct Results {
    pub training_time_seconds: f64,
    pub final_valid_accuracy: f64,
    pub final_valid_loss: f64,
}

#[derive(Serialize)]
pub struct ExperimentSummary {
    pub experiment_name: String,
    pub device: String,
    pub hyperparameters: Hyperparameters,
    pub results: Results,
    pub model_path: String,
}

pub struct ExperimentLogger {
    experiment_name: String,
    artifact_dir: String,
    hyperparameters: Hyperparameters,
    device_info: String,
    start_time: Option<Instant>,
}

impl ExperimentLogger {
    pub fn new(
        experiment_name: impl Into<String>,
        artifact_dir: impl Into<String>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        optimizer: impl Into<String>,
        workers: usize,
        seed: u64,
        device_info: impl Into<String>,
    ) -> Self {
        Self {
            experiment_name: experiment_name.into(),
            artifact_dir: artifact_dir.into(),
            hyperparameters: Hyperparameters {
                epochs,
                batch_size,
                learning_rate,
                optimizer: optimizer.into(),
                workers,
                seed,
            },
            device_info: device_info.into(),
            start_time: None,
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    pub fn complete_and_save(
        &self,
        model_relative_path: &str,
        valid_accuracy: f64,
        valid_loss: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !valid_accuracy.is_finite() || !valid_loss.is_finite() {
            return Err(format!(
                "Validation metrics must be finite: accuracy={valid_accuracy}, loss={valid_loss}"
            )
            .into());
        }

        let elapsed = self
            .start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        let summary = ExperimentSummary {
            experiment_name: self.experiment_name.clone(),
            device: self.device_info.clone(),
            hyperparameters: Hyperparameters {
                epochs: self.hyperparameters.epochs,
                batch_size: self.hyperparameters.batch_size,
                learning_rate: self.hyperparameters.learning_rate,
                optimizer: self.hyperparameters.optimizer.clone(),
                workers: self.hyperparameters.workers,
                seed: self.hyperparameters.seed,
            },
            results: Results {
                training_time_seconds: elapsed,
                final_valid_accuracy: valid_accuracy,
                final_valid_loss: valid_loss,
            },
            model_path: model_relative_path.to_string(),
        };

        let json_data = serde_json::to_string_pretty(&summary)?;
        let artifact_dir = std::path::Path::new(&self.artifact_dir);
        fs::create_dir_all(artifact_dir)?;
        let json_path = artifact_dir.join(format!(
            "{}_{}.json",
            self.experiment_name, self.hyperparameters.epochs
        ));
        let mut file = File::create(&json_path)?;
        file.write_all(json_data.as_bytes())?;

        println!("Valid Accuracy : {:.2}%", valid_accuracy * 100.0);
        println!("Valid Loss     : {:.4}", valid_loss);

        Ok(())
    }
}