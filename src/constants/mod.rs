pub const GIB: usize = 1_073_741_824;

#[derive(Clone)]
pub struct UCIConfig {
    // Cache size of transposition table in bytes
    pub tt_cache_size: usize,

    // Relative path to NN weights
    pub nn_weights: String, 

    // Threads for onnx runtime
    pub n_onnx_threads: usize, 

    // Depth (root=0) up to which to use deep move ordering
    pub deep_move_ordering_depth: u8, 

    // How many moves to not shuffle at beginning of moves vec
    // In general, lazy smp works better with different threads searching
    // different move orders...but we have a nice NN that picks what it thinks
    // are the best moves to search first. So a compromise is to let the NN
    // pick the first N and shuffle the rest.
    pub lazy_smp_shuffle_n: usize,

    // How many parallel threads search root
    pub lazy_smp_parallel_root: usize, 

    // Time allocated for each search - sort of
    // Actually the time each negamax instance has before terminating
    pub search_time: std::time::Duration,

    // The maximun search depth for ponder searches. Regular searches are 
    // limited by time, not depth.
    pub ponder_search_depth: u8,

    // Number of LRU cache entries for ponder cache
    pub ponder_cache_size: usize,
}

impl UCIConfig {
    pub fn default() -> Self {
        Self { 
            tt_cache_size: GIB, 
            nn_weights: String::from("nn.quant.onnx"), 
            n_onnx_threads: 4, 
            deep_move_ordering_depth: 0, 
            lazy_smp_shuffle_n: 20, 
            lazy_smp_parallel_root: 2, 
            search_time: std::time::Duration::from_secs(3), 
            ponder_search_depth: 7, 
            ponder_cache_size: 1000, 
        }
    }

    pub fn help() -> String {
        let options = vec![
            "option name tt_cache_size type spin default 1073741824 min 1 max 1073741824",
            "option name nn_weights type string default \"nn.quant.onnx\"",
            "option name n_onnx_threads type spin default 4 min 1 max 1024",
            "option name deep_move_ordering_depth type spin default 0 min 0 max 255",
            "option name lazy_smp_shuffle_n type spin default 20 min 0 max 1024",
            "option name lazy_smp_parallel_root type spin default 2 min 0 max 64",
            "option name search_time type spin default 3 min 0 max 1000000",
            "option name ponder_search_depth type spin default 7 min 0 max 255",
            "option name ponder_cache_size type spin default 1000 min 1 max 1073741824",
        ];

        options.join("\n")
    }

    /// Retursn true if engine needs to be rebuilt, false otherwise
    pub fn set(&mut self, name: &str, value: &str) -> Result<bool, Box<dyn std::error::Error>> {
        match name {
            "tt_cache_size" => {
                self.tt_cache_size = value.parse()?;
                Ok(true)
            }, 
            "nn_weights" => {
                self.nn_weights = value.to_string();
                Ok(true)
            },
            "n_onnx_threads" => {
                self.n_onnx_threads = value.parse()?;
                Ok(true)
            },
            "deep_move_ordering_depth" => {
                self.deep_move_ordering_depth = value.parse()?;
                Ok(false)
            },
            "lazy_smp_shuffle_n" => {
                self.lazy_smp_shuffle_n = value.parse()?;
                Ok(false)
            },
            "lazy_smp_parallel_root" => {
                self.lazy_smp_parallel_root = value.parse()?;
                Ok(false)
            }
            "search_time" => {
                self.search_time = std::time::Duration::from_secs(value.parse()?);
                Ok(false)
            },
            "ponder_search_depth" => {
                self.ponder_search_depth = value.parse()?; 
                Ok(false)
            },
            "ponder_cache_size" => {
                self.ponder_cache_size = value.parse()?;
                Ok(true)
            }
            _ => {
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "invalid uci option")))
            }
        }
    }
}
