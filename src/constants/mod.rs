pub const GIB: usize = 1_073_741_824;

// Cache size of transposition table in bytes
pub const TT_CACHE_SIZE: usize = GIB;

// Relative path to NN weights
pub const NN_WEIGHTS: &str = "nn.quant.onnx";

// Threads for onnx runtime
pub const N_ONNX_THREADS: usize = 4;

// Depth (root=0) up to which to use deep move ordering
pub const DEEP_MOVE_ORDERING_DEPTH: u8 = 0;

// How many moves to not shuffle at beginning of moves vec
// In general, lazy smp works better with different threads searching
// different move orders...but we have a nice NN that picks what it thinks
// are the best moves to search first. So a compromise is to let the NN
// pick the first N and shuffle the rest.
pub const LAZY_SMP_SHUFFLE_N: usize = 20;

// How many parallel threads search root
pub const LAZY_SMP_PARALLEL_ROOT: usize = 2;

// Time allocated for each search - sort of
// Actually the time each negamax instance has before terminating
pub const SEARCH_TIME: std::time::Duration = std::time::Duration::from_secs(3);

// The maximun search depth for ponder searches. Regular searches are 
// limited by time, not depth.
pub const PONDER_SEARCH_DEPTH: u8 = 7;