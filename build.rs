use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Path to the source file
    let source_path = "nn-training/runs/run-1/nn.quant.onnx";

    // Determine the build output directory (e.g., target/debug or target/release)
    let profile = env::var("PROFILE").expect("PROFILE environment variable is not set");
    let target_dir = env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    let executable_dir = Path::new(&target_dir).join(&profile);

    // Construct the destination path
    let dest_path = executable_dir.join("nn.quant.onnx");

    // Copy the file
    fs::create_dir_all(dest_path.parent().unwrap())
        .expect("Failed to create output directory");
    fs::copy(source_path, &dest_path)
        .expect("Failed to copy nn.quant.onnx");

    println!("cargo:rerun-if-changed={}", source_path);
}