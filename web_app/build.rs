use std::path::Path;

use vergen_gitcl::BuildBuilder;
use vergen_gitcl::CargoBuilder;
use vergen_gitcl::Emitter;
use vergen_gitcl::GitclBuilder;
use vergen_gitcl::RustcBuilder;
use vergen_gitcl::SysinfoBuilder;

fn main() -> Result<
    (),
    Box<dyn std::error::Error>,
> {

    println!("cargo:rustc-env=VERGEN_SYSINFO_KERNEL_VERSION=unknown");

    let mut emitter =
        Emitter::default();

    emitter.idempotent();

    emitter.add_instructions(
        &BuildBuilder::all_build()?,
    )?;

    emitter.add_instructions(
        &CargoBuilder::all_cargo()?,
    )?;

    emitter.add_instructions(
        &GitclBuilder::all_git()?,
    )?;

    emitter.add_instructions(
        &RustcBuilder::all_rustc()?,
    )?;

    emitter.add_instructions(
        &SysinfoBuilder::all_sysinfo()?,
    )?;

    emitter.emit()?;

    // Tokenizer conversion logic
    let tokenizer_json_path =
        if Path::new(
            "assets/tokenizer.json",
        )
        .exists()
        {

            "assets/tokenizer.json"
                .to_string()
        } else if Path::new(
            "../assets/tokenizer.json",
        )
        .exists()
        {

            "../assets/tokenizer.json"
                .to_string()
        } else {

            println!("cargo:warning=tokenizer.json not found, skipping bincode conversion");

            return Ok(());
        };

    let tokenizer_bin_path =
        tokenizer_json_path
            .replace(".json", ".bin");

    if !Path::new(&tokenizer_bin_path)
        .exists()
        || std::fs::metadata(
            &tokenizer_json_path,
        )?
        .modified()?
            > std::fs::metadata(
                &tokenizer_bin_path,
            )?
            .modified()?
    {

        println!(
            "cargo:rerun-if-changed={}",
            tokenizer_json_path
        );

        println!(
            "cargo:warning=Converting \
             {} to bincode...",
            tokenizer_json_path
        );

        // let tokenizer_json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&tokenizer_json_path)?)
        //     .map_err(|e| format!("Failed to parse tokenizer JSON: {}", e))?;

        let mut bytes = std::fs::read(
            &tokenizer_json_path,
        )
        .map_err(|e| {

            format!(
                "Failed to read file: \
                 {}",
                e
            )
        })?;

        let tokenizer_json: serde_json::Value = simd_json::from_slice(&mut bytes)
            .map_err(|e| format!("Failed to parse tokenizer JSON with SIMD: {}", e))?;

        let config = bincode_next::config::legacy();

        let encoded = bincode_next::serde::encode_to_vec(&tokenizer_json, config)
            .map_err(|e| format!("Failed to encode tokenizer Value to bincode: {}", e))?;

        std::fs::write(
            &tokenizer_bin_path,
            encoded,
        )?;

        println!(
            "cargo:warning=Converted \
             tokenizer saved to {}",
            tokenizer_bin_path
        );
    }

    // Model conversion/copy logic
    let model_gguf_path =
        tokenizer_json_path.replace(
            "tokenizer.json",
            "gemma-270m.gguf",
        );

    let model_bin_path =
        tokenizer_json_path.replace(
            "tokenizer.json",
            "llm.bin",
        );

    if Path::new(&model_gguf_path)
        .exists()
    {

        // Tell cargo to rerun if the source model changes
        println!(
            "cargo:rerun-if-changed={}",
            model_gguf_path
        );

        // Copy if destination doesn't exist OR source is newer than destination
        let should_copy = !Path::new(
            &model_bin_path,
        )
        .exists()
            || std::fs::metadata(
                &model_gguf_path,
            )?
            .modified()?
                > std::fs::metadata(
                    &model_bin_path,
                )?
                .modified()?;

        if should_copy {

            println!("cargo:warning=Copying {} to {}...", model_gguf_path, model_bin_path);

            std::fs::copy(&model_gguf_path, &model_bin_path)
                .map_err(|e| format!("Failed to copy model file: {}", e))?;
        }
    } else {

        println!(
            "cargo:warning=gemma-270m.\
             gguf not found at {}, \
             skipping copy",
            model_gguf_path
        );
    }

    Ok(())
}
