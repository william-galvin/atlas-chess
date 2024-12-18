pub const GIB: usize = 1_073_741_824;
// Cache size of transposition table in bytes
pub const TT_CACHE_SIZE: usize = GIB / 1024;

// Threads for onnx runtime
pub const N_ONNX_THREADS: usize = 2;
// Depth (root=0) up to which to use deep move ordering
pub const DEEP_MOVE_ORDERING_DEPTH: u8 = 0;