#[allow(dead_code, reason = "used in binary, compiler warns for library")]
pub const GIB: usize = 1_073_741_824;

#[allow(dead_code, reason = "used in binary, compiler warns for library")]
#[derive(Clone)]
pub struct UCIConfig {
    // Cache size of transposition table in bytes
    pub tt_cache_size: usize,

    // Threads for onnx runtime
    pub n_onnx_threads: usize, 

    // Depth (root=0) up to which to use deep move ordering
    pub deep_move_ordering_depth: u8, 

    // Time allocated for each search - sort of
    // Actually the time each negamax instance has before terminating
    pub search_time: std::time::Duration,

    // The maximun search depth for searches - also 
    // limited by time
    pub search_depth: u8,

    // Number of LRU cache entries for ponder cache
    pub ponder_cache_size: usize,

    // If true, use book
    pub own_book: bool,

    // If true, use lichess syzygy tablebase
    pub tablebase: bool,

    // Max number of moves to look past the horizon
    pub qsearch_depth: u8,

    // Depth at which to stop using static eval for forward pruning
    pub static_forward_pruning_depth: u8, 

    // Max branching factor if forward pruning
    pub static_forward_pruning_branch: usize,

}

#[allow(dead_code, reason = "used in binary, compiler warns for library")]
impl UCIConfig {
    pub fn default() -> Self {
        Self { 
            tt_cache_size: GIB, 
            n_onnx_threads: 4, 
            deep_move_ordering_depth: 0,
            search_time: std::time::Duration::from_secs(3), 
            search_depth: 7, 
            ponder_cache_size: 1000, 
            own_book: true,
            tablebase: true,
            qsearch_depth: 4,
            static_forward_pruning_depth: 3,
            static_forward_pruning_branch: 12,
        }
    }

    pub fn help() -> String {
        let options = [
            "option name tt_cache_size type spin default 1073741824 min 1 max 1073741824",
            "option name n_onnx_threads type spin default 4 min 1 max 1024",
            "option name deep_move_ordering_depth type spin default 0 min 0 max 255",
            "option name lazy_smp_shuffle_n type spin default 20 min 0 max 1024",
            "option name lazy_smp_parallel_root type spin default 2 min 0 max 64",
            "option name search_time type spin default 3 min 0 max 1000000",
            "option name search_depth type spin default 7 min 0 max 255",
            "option name ponder_cache_size type spin default 1000 min 1 max 1073741824",
            "option name OwnBook type button default true",
            "option name tablebase type button default true",
            "option name qsearch_depth type spin default 4",
            "option name static_forward_pruning_depth type spin default 3",
            "option name static_forward_pruning_branch type spin default 10",
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
            "n_onnx_threads" => {
                self.n_onnx_threads = value.parse()?;
                Ok(true)
            },
            "deep_move_ordering_depth" => {
                self.deep_move_ordering_depth = value.parse()?;
                Ok(false)
            },
            "search_time" => {
                self.search_time = std::time::Duration::from_secs(value.parse()?);
                Ok(false)
            },
            "search_depth" => {
                self.search_depth = value.parse()?; 
                Ok(false)
            },
            "ponder_cache_size" => {
                self.ponder_cache_size = value.parse()?;
                Ok(true)
            },
            "OwnBook" => {
                self.own_book = value.parse()?;
                Ok(false)
            },
            "tablebase" => {
                self.tablebase = value.parse()?;
                Ok(false)
            },
            "qsearch_depth" => {
                self.qsearch_depth = value.parse()?; 
                Ok(false)
            },
            "static_forward_pruning_depth" => {
                self.static_forward_pruning_depth = value.parse()?; 
                Ok(false)
            },
            "static_forward_pruning_branch" => {
                self.static_forward_pruning_branch = value.parse()?; 
                Ok(false)
            },
            _ => {
                Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "invalid uci option")))
            }
        }
    }
}
