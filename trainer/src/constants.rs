pub const IMAGE_WIDTH: usize = 28;
pub const IMAGE_HEIGHT: usize = 28;
pub const IMAGE_SIZE: usize = IMAGE_WIDTH * IMAGE_HEIGHT; //784
pub const PIXEL_MAX: f32 = 255.0;
pub const IDX_IMAGES_HEADER_SIZE: usize = 16;
pub const IDX_LABELS_HEADER_SIZE: usize = 8;

//Model configuration
pub const EPOCH_NUMBER: usize = 50;
pub const BATCH_SIZE: usize = 64;
pub const LEARNING_RATE: f64 = 3e-4;
pub const SEED: u64 = 4021;
pub const WORKERS: usize = 4;