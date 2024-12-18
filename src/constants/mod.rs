pub const GIB: usize = 1_073_741_824;
// Cache size of transposition table in bytes
pub const TT_CACHE_SIZE: usize = GIB;

// Relative path to NN weights
pub const NN_WEIGHTS: &str = "nn.quant.onnx";
// Threads for onnx runtime
pub const N_ONNX_THREADS: usize = 4;
// Depth (root=0) up to which to use deep move ordering
pub const DEEP_MOVE_ORDERING_DEPTH: u8 = 0;