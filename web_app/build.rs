use std::path::Path;
use std::sync::Arc;

use num_bigint::BigInt;
use num_bigint::Sign;
use rssn::symbolic::calculus::differentiate;
use rssn::symbolic::calculus::substitute;
use rssn::symbolic::core::Expr;
use rssn::symbolic::simplify_dag::simplify;
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

    emitter.add_instructions(
        &BuildBuilder::all_build()?,
    )?;

    emitter.add_instructions(
        &CargoBuilder::all_cargo()?,
    )?;

    let git = GitclBuilder::default()
        .all()
        .dirty(true)
        .build()?;

    emitter.add_instructions(&git)?;

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

        #[allow(unused_variables)]
        let tokenizer_json: serde_json::Value = simd_json::from_slice(&mut bytes)
            .map_err(|e| format!("Failed to parse tokenizer JSON with SIMD: {}", e))?;

        // Just copy the JSON for now. Bincode + serde_json::Value has Serde(AnyNotSupported) issues.
        std::fs::copy(
            &tokenizer_json_path,
            &tokenizer_bin_path,
        )?;

        println!(
            "cargo:warning=Tokenizer \
             JSON copied to {}",
            tokenizer_bin_path
        );

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

    // Surprise encryption logic
    let surprise_toml_path =
        if Path::new("surprise.toml")
            .exists()
        {

            "surprise.toml"
        } else {

            "../surprise.toml"
        };

    if Path::new(surprise_toml_path)
        .exists()
    {

        println!(
            "cargo:rerun-if-changed={}",
            surprise_toml_path
        );

        let content =
            std::fs::read_to_string(
                surprise_toml_path,
            )?;

        #[derive(
            serde::Deserialize,
        )]

        struct Surprise {
            key: String,
            payload: String,
        }

        let surprise: Surprise =
            toml::from_str(&content)?;

        // Parse key
        let (_, mut key_expr) = rssn::input::parser::parse_expr(&surprise.key)
            .map_err(|e| format!("Failed to parse key: {}", e))?;

        key_expr =
            force_bigint(key_expr);

        // 5th derivative
        let mut deriv = key_expr;

        for _ in 0 .. 5 {

            deriv = differentiate(
                &deriv,
                "x",
            );
        }

        deriv = simplify(&deriv)
            .to_ast()
            .map_err(|e| {
                format!(
                    "to_ast failed: {}",
                    e
                )
            })?;

        deriv = force_bigint(deriv);

        // Convert payload to BigInt
        let payload_num =
            BigInt::from_bytes_be(
                Sign::Plus,
                surprise
                    .payload
                    .as_bytes(),
            );

        // Evaluate at P
        // result = f(P)
        let eval_expr = substitute(
            &deriv,
            "x",
            &Expr::BigInt(payload_num),
        );

        let result_expr = simplify(
            &eval_expr,
        )
        .to_ast()
        .map_err(|e| {
            format!(
                "to_ast failed: {}",
                e
            )
        })?;

        let result_expr =
            force_bigint(result_expr);

        // Serialize
        let assets_dir =
            if Path::new("assets")
                .exists()
            {

                "assets"
            } else {

                "../assets"
            };

        let surprise_bin_path =
            Path::new(assets_dir)
                .join("surprise.bin");

        let encoded: Vec<u8> = bincode_next::serde::encode_to_vec(&result_expr, bincode_next::config::standard())
            .map_err(|e| format!("Failed to encode: {}", e))?;

        std::fs::write(
            &surprise_bin_path,
            encoded,
        )?;

        println!(
            "cargo:warning=Surprise \
             encrypted and saved to \
             {:?}",
            surprise_bin_path
        );
    }

    Ok(())
}

fn force_bigint(expr: Expr) -> Expr {

    match expr {
        | Expr::Constant(f)
            if f.fract() == 0.0 =>
        {

            // Use i64 for small ones, or if we had num-traits we'd use from_f64
            // For CAS coefficients, it's usually small.
            Expr::BigInt(num_bigint::BigInt::from(f as i64))
        },
        | Expr::Add(a, b) => {
            Expr::Add(
                Arc::new(force_bigint(
                    a.as_ref().clone(),
                )),
                Arc::new(force_bigint(
                    b.as_ref().clone(),
                )),
            )
        },
        | Expr::Sub(a, b) => {
            Expr::Sub(
                Arc::new(force_bigint(
                    a.as_ref().clone(),
                )),
                Arc::new(force_bigint(
                    b.as_ref().clone(),
                )),
            )
        },
        | Expr::Mul(a, b) => {
            Expr::Mul(
                Arc::new(force_bigint(
                    a.as_ref().clone(),
                )),
                Arc::new(force_bigint(
                    b.as_ref().clone(),
                )),
            )
        },
        | Expr::Div(a, b) => {
            Expr::Div(
                Arc::new(force_bigint(
                    a.as_ref().clone(),
                )),
                Arc::new(force_bigint(
                    b.as_ref().clone(),
                )),
            )
        },
        | Expr::Power(a, b) => {
            Expr::Power(
                Arc::new(force_bigint(
                    a.as_ref().clone(),
                )),
                Arc::new(force_bigint(
                    b.as_ref().clone(),
                )),
            )
        },
        | Expr::Neg(a) => {
            Expr::Neg(Arc::new(
                force_bigint(
                    a.as_ref().clone(),
                ),
            ))
        },
        | Expr::AddList(list) => {
            Expr::AddList(
                list.into_iter()
                    .map(force_bigint)
                    .collect(),
            )
        },
        | Expr::MulList(list) => {
            Expr::MulList(
                list.into_iter()
                    .map(force_bigint)
                    .collect(),
            )
        },
        | _ => expr,
    }
}
