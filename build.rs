use std::env;
use std::fs;
use std::path::Path;

fn main() {
    for (source_path, dest_path) in vec![
        ("nn-training/runs/run-1/nn.quant.onnx", "nn.quant.onnx"), 
        ("opening_book.csv", "opening_book.csv"),
    ] {
        // Determine the build output directory (e.g., target/debug or target/release)
        let profile = env::var("PROFILE").expect("PROFILE environment variable is not set");
        let target_dir = env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
        let executable_dir = Path::new(&target_dir).join(&profile);

        // Construct the destination path
        let dest_path = executable_dir.join(dest_path);

        // Copy the file
        fs::create_dir_all(dest_path.parent().unwrap())
            .expect("Failed to create output directory");
        fs::copy(source_path, &dest_path)
            .expect("Failed to copy");
    }
}