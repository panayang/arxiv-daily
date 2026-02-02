// Copyright 2025 Xinyu Yang
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

        // Convert payload to BigInt in chunks (1 byte per chunk for maximum safety)
        let payload_bytes = surprise
            .payload
            .as_bytes();

        let chunk_size = 1;

        let mut results = Vec::new();

        // Context-dependent encryption to prevent frequency analysis
        // Initialize context with a salt derived from the key length and first chars
        let mut context = BigInt::from(
            0x5A5A5A5A_u32,
        );

        if !surprise
            .key
            .is_empty()
        {

            // Mix in key-dependent salt
            for b in surprise
                .key
                .as_bytes()
                .iter()
                .take(10)
            {

                context = (&context
                    * BigInt::from(31))
                    + BigInt::from(
                        *b as u64,
                    );
            }

            context = context
                % BigInt::from(
                    0xFFFFFFFF_u64,
                );
        }

        for (index, chunk) in
            payload_bytes
                .chunks(chunk_size)
                .enumerate()
        {

            // Prepend a 1 byte to distinguish from trailing zeros and keep x > 255
            let mut chunk_with_sentinel =
                vec![1u8];

            chunk_with_sentinel
                .extend_from_slice(
                    chunk,
                );

            let p = BigInt::from_bytes_be(Sign::Plus, &chunk_with_sentinel);

            // Mix context into plaintext to make encryption position-dependent
            let position_factor =
                BigInt::from(
                    (index + 1) as u64,
                );

            // Non-linear mixing: p + ((context * pos) % large_prime)
            // This makes p_mixed much larger and harder to pre-calculate
            let context_mod = (&context
                * &position_factor)
                % BigInt::from(
                    0xFFFFFF_u64,
                );

            let p_mixed =
                &p + &context_mod;

            if index == 0 {

                println!("cargo:warning=Chunk 0: p={}, context={}, pos_factor={}, context_mod={}, p_mixed={}", p, context, position_factor, context_mod, p_mixed);
            }

            // Evaluate at P_mixed (context-dependent plaintext)
            let eval_expr = substitute(
                &deriv,
                "x",
                &Expr::BigInt(
                    p_mixed.clone(),
                ),
            );

            let r =
                simplify(&eval_expr)
                    .to_ast()
                    .map_err(|e| {

                        format!(
                    "to_ast failed: {}",
                    e
                )
                    })?;

            // Aggressively force evaluation of numeric parts
            let evaled =
                force_bigint_eval(r);

            // Update context for next iteration
            // Must match decryption exactly!
            let cipher_val_bigint = match &evaled {
                Expr::BigInt(n) => n.clone(),
                Expr::Constant(f) => BigInt::from(f.round() as i64),
                Expr::Rational(r) => r.to_integer(),
                _ => BigInt::from(0),
            };

            context = (&context
                * BigInt::from(31))
                + &cipher_val_bigint
                + &position_factor;

            context = context
                % BigInt::from(
                    0xFFFFFFFF_u64,
                );

            results.push(evaled);
        }

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

        let encoded: Vec<u8> = bincode_next::serde::encode_to_vec(&results, bincode_next::config::standard())
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

fn force_bigint(
    root_expr: Expr
) -> Expr {

    // Stack-based iterative approach to avoid recursion overflow
    use num_traits::Signed;
    use num_traits::ToPrimitive;

    let root_expr_clone =
        root_expr.clone();

    // Helper to extract float value for approximate evaluation
    let try_eval_float =
        |e: &Expr| -> Option<f64> {

            match e {
                | Expr::Constant(f) => {
                    Some(*f)
                },
                | Expr::BigInt(b) => {
                    b.to_f64()
                },
                | _ => None,
            }
        };

    let mut visit_stack =
        vec![(root_expr, false)];

    let mut output_stack: Vec<Expr> =
        Vec::new();

    while let Some((expr, visited)) =
        visit_stack.pop()
    {

        if visited {

            // Reconstruct node from children on output_stack
            match expr {
                Expr::Add(_, _) => {
                    let rhs = output_stack.pop().unwrap();
                    let lhs = output_stack.pop().unwrap();
                    output_stack.push(Expr::Add(Arc::new(lhs), Arc::new(rhs)));
                },
                Expr::Sub(_, _) => {
                    let rhs = output_stack.pop().unwrap();
                    let lhs = output_stack.pop().unwrap();
                    output_stack.push(Expr::Sub(Arc::new(lhs), Arc::new(rhs)));
                },
                Expr::Mul(_, _) => {
                    let rhs = output_stack.pop().unwrap();
                    let lhs = output_stack.pop().unwrap();
                    output_stack.push(Expr::Mul(Arc::new(lhs), Arc::new(rhs)));
                },
                Expr::Div(_, _) => {
                    let rhs = output_stack.pop().unwrap();
                    let lhs = output_stack.pop().unwrap();
                    output_stack.push(Expr::Div(Arc::new(lhs), Arc::new(rhs)));
                },
                Expr::Power(_, _) => {
                    let rhs = output_stack.pop().unwrap();
                    let lhs = output_stack.pop().unwrap();
                    output_stack.push(Expr::Power(Arc::new(lhs), Arc::new(rhs)));
                },
                Expr::LogBase(_, _) => {
                    let rhs = output_stack.pop().unwrap();
                    let lhs = output_stack.pop().unwrap();
                    output_stack.push(Expr::LogBase(Arc::new(lhs), Arc::new(rhs)));
                },
                Expr::Neg(_) => {
                    let inner = output_stack.pop().unwrap();
                    output_stack.push(Expr::Neg(Arc::new(inner)));
                },
                Expr::Sin(_) => {
                    let v = output_stack.pop().unwrap();
                    if let Some(f) = try_eval_float(&v).map(|x| x.sin()) {
                        if (f - f.round()).abs() < 1e-6 {
                            output_stack.push(Expr::BigInt(BigInt::from(f.round() as i64)));
                        } else {
                            output_stack.push(Expr::Constant(f));
                        }
                    } else {
                        output_stack.push(Expr::Sin(Arc::new(v)));
                    }
                },
                Expr::Cos(_) => {
                    let v = output_stack.pop().unwrap();
                    if let Some(f) = try_eval_float(&v).map(|x| x.cos()) {
                        if (f - f.round()).abs() < 1e-6 {
                            output_stack.push(Expr::BigInt(BigInt::from(f.round() as i64)));
                        } else {
                            output_stack.push(Expr::Constant(f));
                        }
                    } else {
                        output_stack.push(Expr::Cos(Arc::new(v)));
                    }
                },
                Expr::Tan(_) => {
                    let v = output_stack.pop().unwrap();
                    if let Some(f) = try_eval_float(&v).map(|x| x.tan()) {
                        if (f - f.round()).abs() < 1e-6 {
                            output_stack.push(Expr::BigInt(BigInt::from(f.round() as i64)));
                        } else {
                            output_stack.push(Expr::Constant(f));
                        }
                    } else {
                        output_stack.push(Expr::Tan(Arc::new(v)));
                    }
                },
                Expr::Exp(_) => {
                    let v = output_stack.pop().unwrap();
                    if let Some(f) = try_eval_float(&v).map(|x| x.exp()) {
                        if (f - f.round()).abs() < 1e-6 {
                            output_stack.push(Expr::BigInt(BigInt::from(f.round() as i64)));
                        } else {
                            output_stack.push(Expr::Constant(f));
                        }
                    } else {
                        output_stack.push(Expr::Exp(Arc::new(v)));
                    }
                },
                Expr::Log(_) => {
                    let v = output_stack.pop().unwrap();
                    if let Some(f) = try_eval_float(&v).map(|x| x.ln()) {
                        if (f - f.round()).abs() < 1e-6 {
                            output_stack.push(Expr::BigInt(BigInt::from(f.round() as i64)));
                        } else {
                            output_stack.push(Expr::Constant(f));
                        }
                    } else {
                        output_stack.push(Expr::Log(Arc::new(v)));
                    }
                },
                Expr::Abs(_) => {
                    let v = output_stack.pop().unwrap();
                    match v {
                        Expr::BigInt(n) => output_stack.push(Expr::BigInt(n.abs())),
                        Expr::Constant(f) => output_stack.push(Expr::Constant(f.abs())),
                        _ => output_stack.push(Expr::Abs(Arc::new(v))),
                    }
                },
                Expr::Sqrt(_) => {
                    let v = output_stack.pop().unwrap();
                    if let Some(f) = try_eval_float(&v).map(|x| x.sqrt()) {
                        if (f - f.round()).abs() < 1e-6 {
                            output_stack.push(Expr::BigInt(BigInt::from(f.round() as i64)));
                        } else {
                            output_stack.push(Expr::Constant(f));
                        }
                    } else {
                        output_stack.push(Expr::Sqrt(Arc::new(v)));
                    }
                },
                Expr::AddList(list) => {
                    let mut new_list = Vec::with_capacity(list.len());
                    for _ in 0..list.len() {
                        new_list.push(output_stack.pop().unwrap());
                    }
                    new_list.reverse();
                    output_stack.push(Expr::AddList(new_list));
                },
                Expr::MulList(list) => {
                    let mut new_list = Vec::with_capacity(list.len());
                    for _ in 0..list.len() {
                        new_list.push(output_stack.pop().unwrap());
                    }
                    new_list.reverse();
                    output_stack.push(Expr::MulList(new_list));
                },
                // Leaves are pushed directly in the else block
                _ => unreachable!("Leaves or unhandled types should be processed before marking visited. Type: {:?}", std::mem::discriminant(&expr)),
            }
        } else {

            // Process node
            match expr {
                | Expr::Constant(f)
                    if f.fract()
                        == 0.0 =>
                {

                    output_stack.push(Expr::BigInt(num_bigint::BigInt::from(f as i64)));
                },
                // Other leaves just pass through
                | Expr::Constant(_)
                | Expr::BigInt(_)
                | Expr::Variable(_)
                | Expr::Rational(_) => {

                    output_stack
                        .push(expr);
                },
                // Recursive cases: push visited parent, then children (reverse order for stack)
                | Expr::Add(
                    ref a,
                    ref b,
                )
                | Expr::Sub(
                    ref a,
                    ref b,
                )
                | Expr::Mul(
                    ref a,
                    ref b,
                )
                | Expr::Div(
                    ref a,
                    ref b,
                )
                | Expr::Power(
                    ref a,
                    ref b,
                )
                | Expr::LogBase(
                    ref a,
                    ref b,
                ) => {

                    visit_stack.push((
                        expr.clone(),
                        true,
                    ));

                    visit_stack.push((
                        b.as_ref()
                            .clone(),
                        false,
                    ));

                    visit_stack.push((
                        a.as_ref()
                            .clone(),
                        false,
                    ));
                },
                | Expr::Neg(ref a)
                | Expr::Sin(ref a)
                | Expr::Cos(ref a)
                | Expr::Tan(ref a)
                | Expr::Exp(ref a)
                | Expr::Log(ref a)
                | Expr::Abs(ref a)
                | Expr::Sqrt(ref a) => {

                    visit_stack.push((
                        expr.clone(),
                        true,
                    ));

                    visit_stack.push((
                        a.as_ref()
                            .clone(),
                        false,
                    ));
                },
                | Expr::AddList(
                    ref list,
                )
                | Expr::MulList(
                    ref list,
                ) => {

                    visit_stack.push((
                        expr.clone(),
                        true,
                    ));

                    for item in list
                        .iter()
                        .rev()
                    {

                        visit_stack
                            .push((
                            item.clone(
                            ),
                            false,
                        ));
                    }
                },
                // Leaves pass through
                | _ => {

                    output_stack
                        .push(expr);
                },
            }
        }
    }

    output_stack
        .pop()
        .unwrap_or(root_expr_clone)
}

fn force_bigint_eval(
    root_expr: Expr
) -> Expr {

    use std::sync::Arc;

    use num_traits::Signed;
    use num_traits::ToPrimitive;
    use num_traits::Zero;

    let root = root_expr
        .to_ast()
        .unwrap_or(root_expr.clone());

    let try_eval_float =
        |e: &Expr| -> Option<f64> {

            match e {
                | Expr::Constant(f) => {
                    Some(*f)
                },
                | Expr::BigInt(b) => {
                    b.to_f64()
                },
                | _ => None,
            }
        };

    let mut visit_stack =
        vec![(root, false)];

    let mut output_stack: Vec<Expr> =
        Vec::new();

    while let Some((expr, visited)) =
        visit_stack.pop()
    {

        if visited {

            match expr {
                | Expr::Add(_, _) => {

                    let rhs =
                        output_stack
                            .pop()
                            .unwrap();

                    let lhs =
                        output_stack
                            .pop()
                            .unwrap();

                    match (lhs, rhs) {
                        (Expr::BigInt(la), Expr::BigInt(lb)) => output_stack.push(Expr::BigInt(la + lb)),
                        (Expr::Constant(la), Expr::Constant(lb)) => output_stack.push(Expr::Constant(la + lb)),
                        (Expr::BigInt(la), Expr::Constant(lb)) => {
                            if la.abs() > BigInt::from(1_000_000_000_000_i64) {
                                output_stack.push(Expr::Add(Arc::new(Expr::BigInt(la)), Arc::new(Expr::Constant(lb))));
                            } else {
                                output_stack.push(Expr::Constant(la.to_f64().unwrap_or(0.0) + lb));
                            }
                        },
                        (Expr::Constant(la), Expr::BigInt(lb)) => {
                            if lb.abs() > BigInt::from(1_000_000_000_000_i64) {
                                output_stack.push(Expr::Add(Arc::new(Expr::Constant(la)), Arc::new(Expr::BigInt(lb))));
                            } else {
                                output_stack.push(Expr::Constant(la + lb.to_f64().unwrap_or(0.0)));
                            }
                        },
                        (l, r) => output_stack.push(Expr::Add(Arc::new(l), Arc::new(r))),
                    }
                },
                | Expr::Sub(_, _) => {

                    let rhs =
                        output_stack
                            .pop()
                            .unwrap();

                    let lhs =
                        output_stack
                            .pop()
                            .unwrap();

                    match (lhs, rhs) {
                        (Expr::BigInt(la), Expr::BigInt(lb)) => output_stack.push(Expr::BigInt(la - lb)),
                        (Expr::Constant(la), Expr::Constant(lb)) => output_stack.push(Expr::Constant(la - lb)),
                        (Expr::BigInt(la), Expr::Constant(lb)) => {
                            if la.abs() > BigInt::from(1_000_000_000_000_i64) {
                                output_stack.push(Expr::Sub(Arc::new(Expr::BigInt(la)), Arc::new(Expr::Constant(lb))));
                            } else {
                                output_stack.push(Expr::Constant(la.to_f64().unwrap_or(0.0) - lb));
                            }
                        },
                        (Expr::Constant(la), Expr::BigInt(lb)) => {
                            if lb.abs() > BigInt::from(1_000_000_000_000_i64) {
                                output_stack.push(Expr::Sub(Arc::new(Expr::Constant(la)), Arc::new(Expr::BigInt(lb))));
                            } else {
                                output_stack.push(Expr::Constant(la - lb.to_f64().unwrap_or(0.0)));
                            }
                        },
                        (l, r) => output_stack.push(Expr::Sub(Arc::new(l), Arc::new(r))),
                    }
                },
                | Expr::Mul(_, _) => {

                    let rhs =
                        output_stack
                            .pop()
                            .unwrap();

                    let lhs =
                        output_stack
                            .pop()
                            .unwrap();

                    match (lhs, rhs) {
                        (Expr::BigInt(la), Expr::BigInt(lb)) => output_stack.push(Expr::BigInt(la * lb)),
                        (Expr::Constant(la), Expr::Constant(lb)) => output_stack.push(Expr::Constant(la * lb)),
                        (Expr::BigInt(la), Expr::Constant(lb)) => {
                            if la.abs() > BigInt::from(1_000_000_000_000_i64) {
                                output_stack.push(Expr::Mul(Arc::new(Expr::BigInt(la)), Arc::new(Expr::Constant(lb))));
                            } else {
                                output_stack.push(Expr::Constant(la.to_f64().unwrap_or(0.0) * lb));
                            }
                        },
                        (Expr::Constant(la), Expr::BigInt(lb)) => {
                            if lb.abs() > BigInt::from(1_000_000_000_000_i64) {
                                output_stack.push(Expr::Mul(Arc::new(Expr::Constant(la)), Arc::new(Expr::BigInt(lb))));
                            } else {
                                output_stack.push(Expr::Constant(la * lb.to_f64().unwrap_or(0.0)));
                            }
                        },
                        (l, r) => output_stack.push(Expr::Mul(Arc::new(l), Arc::new(r))),
                    }
                },
                | Expr::Div(_, _) => {

                    let rhs =
                        output_stack
                            .pop()
                            .unwrap();

                    let lhs =
                        output_stack
                            .pop()
                            .unwrap();

                    match (lhs, rhs) {
                        (Expr::BigInt(la), Expr::BigInt(lb)) if !lb.is_zero() && &la % &lb == BigInt::from(0) => output_stack.push(Expr::BigInt(la / lb)),
                        (Expr::Constant(la), Expr::Constant(lb)) => output_stack.push(Expr::Constant(la / lb)),
                        (Expr::BigInt(la), Expr::Constant(lb)) => output_stack.push(Expr::Constant(la.to_f64().unwrap_or(0.0) / lb)),
                        (Expr::Constant(la), Expr::BigInt(lb)) => output_stack.push(Expr::Constant(la / lb.to_f64().unwrap_or(0.0))),
                        (l, r) => output_stack.push(Expr::Div(Arc::new(l), Arc::new(r))),
                    }
                },
                | Expr::Power(_, _) => {

                    let rhs =
                        output_stack
                            .pop()
                            .unwrap();

                    let lhs =
                        output_stack
                            .pop()
                            .unwrap();

                    match (lhs, rhs) {
                        (Expr::BigInt(base), Expr::BigInt(exp)) => {
                            if let Some(e) = exp.to_u32() {
                                output_stack.push(Expr::BigInt(base.pow(e)));
                            } else if exp == BigInt::from(0) {
                                output_stack.push(Expr::BigInt(BigInt::from(1)));
                            } else {
                                output_stack.push(Expr::Power(Arc::new(Expr::BigInt(base)), Arc::new(Expr::BigInt(exp))));
                            }
                        }
                        (Expr::Constant(b), Expr::Constant(e)) => output_stack.push(Expr::Constant(b.powf(e))),
                        (Expr::BigInt(b), Expr::Constant(e)) => output_stack.push(Expr::Constant(b.to_f64().unwrap_or(0.0).powf(e))),
                        (Expr::Constant(b), Expr::BigInt(e)) => output_stack.push(Expr::Constant(b.powf(e.to_f64().unwrap_or(0.0)))),
                        (l, r) => output_stack.push(Expr::Power(Arc::new(l), Arc::new(r))),
                    }
                },
                | Expr::Neg(_) => {

                    let inner =
                        output_stack
                            .pop()
                            .unwrap();

                    match inner {
                        Expr::BigInt(n) => output_stack.push(Expr::BigInt(-n)),
                        other => output_stack.push(Expr::Neg(Arc::new(other))),
                    }
                },
                | Expr::Sin(_) => {

                    let v =
                        output_stack
                            .pop()
                            .unwrap();

                    if let Some(f) =
                        try_eval_float(
                            &v,
                        )
                        .map(|x| {
                            x.sin()
                        })
                    {

                        if (f - f
                            .round())
                        .abs()
                            < 1e-6
                        {

                            output_stack.push(Expr::BigInt(BigInt::from(f.round() as i64)));
                        } else {

                            output_stack.push(Expr::Constant(f));
                        }
                    } else {

                        output_stack.push(Expr::Sin(Arc::new(v)));
                    }
                },
                | Expr::Cos(_) => {

                    let v =
                        output_stack
                            .pop()
                            .unwrap();

                    if let Some(f) =
                        try_eval_float(
                            &v,
                        )
                        .map(|x| {
                            x.cos()
                        })
                    {

                        if (f - f
                            .round())
                        .abs()
                            < 1e-6
                        {

                            output_stack.push(Expr::BigInt(BigInt::from(f.round() as i64)));
                        } else {

                            output_stack.push(Expr::Constant(f));
                        }
                    } else {

                        output_stack.push(Expr::Cos(Arc::new(v)));
                    }
                },
                | Expr::Tan(_) => {

                    let v =
                        output_stack
                            .pop()
                            .unwrap();

                    if let Some(f) =
                        try_eval_float(
                            &v,
                        )
                        .map(|x| {
                            x.tan()
                        })
                    {

                        if (f - f
                            .round())
                        .abs()
                            < 1e-6
                        {

                            output_stack.push(Expr::BigInt(BigInt::from(f.round() as i64)));
                        } else {

                            output_stack.push(Expr::Constant(f));
                        }
                    } else {

                        output_stack.push(Expr::Tan(Arc::new(v)));
                    }
                },
                | Expr::Exp(_) => {

                    let v =
                        output_stack
                            .pop()
                            .unwrap();

                    if let Some(f) =
                        try_eval_float(
                            &v,
                        )
                        .map(|x| {
                            x.exp()
                        })
                    {

                        if (f - f
                            .round())
                        .abs()
                            < 1e-6
                        {

                            output_stack.push(Expr::BigInt(BigInt::from(f.round() as i64)));
                        } else {

                            output_stack.push(Expr::Constant(f));
                        }
                    } else {

                        output_stack.push(Expr::Exp(Arc::new(v)));
                    }
                },
                | Expr::Log(_) => {

                    let v =
                        output_stack
                            .pop()
                            .unwrap();

                    if let Some(f) =
                        try_eval_float(
                            &v,
                        )
                        .map(|x| x.ln())
                    {

                        if (f - f
                            .round())
                        .abs()
                            < 1e-6
                        {

                            output_stack.push(Expr::BigInt(BigInt::from(f.round() as i64)));
                        } else {

                            output_stack.push(Expr::Constant(f));
                        }
                    } else {

                        output_stack.push(Expr::Log(Arc::new(v)));
                    }
                },
                | Expr::Abs(_) => {

                    let v =
                        output_stack
                            .pop()
                            .unwrap();

                    if let Some(f) =
                        try_eval_float(
                            &v,
                        )
                    {

                        output_stack.push(Expr::Constant(f.abs()));
                    } else {

                        match v {
                            Expr::BigInt(n) => output_stack.push(Expr::BigInt(n.abs())),
                            _ => output_stack.push(Expr::Abs(Arc::new(v))),
                        }
                    }
                },
                | Expr::Sqrt(_) => {

                    let v =
                        output_stack
                            .pop()
                            .unwrap();

                    if let Some(f) =
                        try_eval_float(
                            &v,
                        )
                        .map(|x| {
                            x.sqrt()
                        })
                    {

                        if (f - f
                            .round())
                        .abs()
                            < 1e-6
                        {

                            output_stack.push(Expr::BigInt(BigInt::from(f.round() as i64)));
                        } else {

                            output_stack.push(Expr::Constant(f));
                        }
                    } else {

                        output_stack.push(Expr::Sqrt(Arc::new(v)));
                    }
                },
                | Expr::LogBase(
                    _,
                    _,
                ) => {

                    let rhs =
                        output_stack
                            .pop()
                            .unwrap();

                    let lhs =
                        output_stack
                            .pop()
                            .unwrap();

                    output_stack.push(
                        Expr::LogBase(
                            Arc::new(
                                lhs,
                            ),
                            Arc::new(
                                rhs,
                            ),
                        ),
                    );
                },
                | Expr::AddList(
                    list,
                ) => {

                    let len =
                        list.len();

                    let mut new_list = Vec::with_capacity(len);

                    for _ in 0 .. len {

                        new_list.push(output_stack.pop().unwrap());
                    }

                    new_list.reverse();

                    output_stack.push(
                        Expr::AddList(
                            new_list,
                        ),
                    );
                },
                | Expr::MulList(
                    list,
                ) => {

                    let len =
                        list.len();

                    let mut new_list = Vec::with_capacity(len);

                    for _ in 0 .. len {

                        new_list.push(output_stack.pop().unwrap());
                    }

                    new_list.reverse();

                    output_stack.push(
                        Expr::MulList(
                            new_list,
                        ),
                    );
                },
                | _ => {

                    output_stack
                        .push(expr);
                },
            }
        } else {

            let processed_leaf = match &expr {
                Expr::Constant(f) if f.fract() == 0.0 => Some(Expr::BigInt(BigInt::from(*f as i64))),
                Expr::Rational(r) if r.is_integer() => Some(Expr::BigInt(r.to_integer())),
                Expr::Constant(_) | Expr::Rational(_) | Expr::Variable(_) | Expr::BigInt(_) => Some(expr.clone()),
                _ => None,
            };

            if let Some(leaf) =
                processed_leaf
            {

                output_stack.push(leaf);

                continue;
            }

            match &expr {
                | Expr::Add(a, b)
                | Expr::Sub(a, b)
                | Expr::Mul(a, b)
                | Expr::Div(a, b)
                | Expr::Power(a, b)
                | Expr::LogBase(
                    a,
                    b,
                ) => {

                    visit_stack.push((
                        expr.clone(),
                        true,
                    ));

                    visit_stack.push((
                        b.to_ast()
                            .unwrap_or(
                            b.as_ref()
                                .clone(
                                ),
                        ),
                        false,
                    ));

                    visit_stack.push((
                        a.to_ast()
                            .unwrap_or(
                            a.as_ref()
                                .clone(
                                ),
                        ),
                        false,
                    ));
                },
                | Expr::Neg(a)
                | Expr::Sin(a)
                | Expr::Cos(a)
                | Expr::Tan(a)
                | Expr::Exp(a)
                | Expr::Log(a)
                | Expr::Abs(a)
                | Expr::Sqrt(a) => {

                    visit_stack.push((
                        expr.clone(),
                        true,
                    ));

                    visit_stack.push((
                        a.to_ast()
                            .unwrap_or(
                            a.as_ref()
                                .clone(
                                ),
                        ),
                        false,
                    ));
                },
                | Expr::AddList(
                    list,
                )
                | Expr::MulList(
                    list,
                ) => {

                    visit_stack.push((
                        expr.clone(),
                        true,
                    ));

                    for item in list
                        .iter()
                        .rev()
                    {

                        visit_stack.push((item.to_ast().unwrap_or(item.clone()), false));
                    }
                },
                | _ => {

                    output_stack
                        .push(expr);
                },
            }
        }
    }

    output_stack
        .pop()
        .unwrap_or(root_expr)
}
