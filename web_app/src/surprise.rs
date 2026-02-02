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

#![allow(unused_imports)]

use leptos::prelude::*;
use num_bigint::BigInt;
#[cfg(feature = "ssr")]
use rssn::symbolic::core::Expr;

#[cfg(feature = "ssr")]

pub async fn decrypt_surprise_logic(
    user_key: &str
) -> Result<String, String> {

    let user_key = user_key.trim();

    use std::path::Path;

    use num_bigint::BigInt;
    use rssn::symbolic::calculus::differentiate;
    use rssn::symbolic::core::Expr;
    use rssn::symbolic::simplify_dag::simplify;
    use rssn::symbolic::solve::solve;

    let (_, mut key_expr) = rssn::input::parser::parse_expr(user_key)
        .map_err(|e| format!("Invalid key format: {}", e))?;

    key_expr = force_bigint(key_expr);

    log::info!(
        "Parsed and forced key_expr: \
         {}",
        key_expr
    );

    let assets_dirs = [
        "assets",
        "../assets",
        "public",
        "../public",
    ];

    let mut bin_path = None;

    for dir in assets_dirs {

        let path = Path::new(dir)
            .join("surprise.bin");

        if path.exists() {

            bin_path = Some(path);

            break;
        }
    }

    let path = bin_path.ok_or(
        "Surprise data (surprise.bin) \
         not found",
    )?;

    let bytes =
        std::fs::read(path.clone())
            .map_err(|e| {

                format!(
                    "Failed to read \
                     surprise data: {}",
                    e
                )
            })?;

    log::info!(
        "Read {} bytes from {:?}",
        bytes.len(),
        path
    );

    let decoded: (Vec<Expr>, usize) = bincode_next::serde::decode_from_slice(&bytes, bincode_next::config::standard())
        .map_err(|e| format!("Failed to deserialize: {}", e))?;

    let encrypted_exprs = decoded.0;

    log::info!(
        "Read {} encrypted chunks",
        encrypted_exprs.len()
    );

    let mut deriv = key_expr;

    for _ in 0 .. 5 {

        deriv =
            differentiate(&deriv, "x");
    }

    deriv = simplify(&deriv)
        .to_ast()
        .unwrap_or_else(|_| {

            simplify(&deriv)
        });

    deriv = force_bigint(deriv);

    log::info!(
        "Derivative expr: {}",
        deriv
    );

    let mut full_message_bytes =
        Vec::new();

    // Initialize context with a salt derived from the user_key
    // Must match build.rs exactly!
    let mut context =
        BigInt::from(0x5A5A5A5A_u32);

    if !user_key.is_empty() {

        for b in user_key
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

    for (i, encrypted_expr) in
        encrypted_exprs
            .into_iter()
            .enumerate()
    {

        let mut real_roots = Vec::new();

        let position_factor =
            BigInt::from(
                (i + 1) as u64,
            );

        let mixing_val = (&context
            * &position_factor)
            % BigInt::from(
                0xFFFFFF_u64,
            );

        if i == 0 {

            log::info!(
                "Chunk 0: context={}, \
                 pos_factor={}, \
                 mixing_val={}, \
                 target={:?}",
                context,
                position_factor,
                mixing_val,
                encrypted_expr
            );
        }

        // Strategy 1: BigInt-aware solver (exact for large integers, handles up to quadratic)
        if let Some(root) =
            try_solve_bigint(
                &deriv,
                &encrypted_expr,
                &mixing_val,
            )
        {

            real_roots.push(root);
        } else {

            // Strategy 2: Standard solve (fallback)
            let eq = Expr::new_sub(
                deriv.clone(),
                encrypted_expr.clone(),
            );

            let solutions =
                solve(&eq, "x");

            for sol in solutions {

                let s = simplify(&sol)
                    .to_ast()
                    .unwrap_or_else(
                        |_| {

                            simplify(
                                &sol,
                            )
                        },
                    );

                match s {
                    | Expr::BigInt(n) => real_roots.push(n.clone()),
                    | Expr::Rational(r) => {
                        if r.is_integer() {
                            real_roots.push(r.to_integer());
                        }
                    },
                    | Expr::Constant(f) => {
                        if (f - f.round()).abs() < 1e-6 {
                            if f.abs() < (i64::MAX as f64) {
                                real_roots.push(BigInt::from(f.round() as i64));
                            }
                        }
                    },
                    | _ => {},
                }
            }
        }

        if let Some(p_mixed) =
            real_roots.first()
        {

            // Reverse the context mixing to get original plaintext
            // Must match build.rs: p_mixed = p + ((context * position) % 0xFFFFFF)
            let target_num =
                p_mixed - &mixing_val;

            let mut bytes = target_num
                .to_bytes_be()
                .1;

            if !bytes.is_empty() {

                bytes.remove(0); // Remove sentinel byte
                full_message_bytes
                    .extend(bytes);
            }

            // Update context to match encryption's context update
            // Must match build.rs exactly!
            let cipher_val_bigint = match &encrypted_expr {
                Expr::BigInt(n) => n.clone(),
                Expr::Constant(f) => {
                    // Convert float to BigInt (round to nearest integer)
                    BigInt::from(f.round() as i64)
                },
                Expr::Rational(r) => r.to_integer(),
                _ => {
                    log::warn!("Unexpected cipher_val type: {:?}, using 0", encrypted_expr);
                    BigInt::from(0)
                }
            };

            context = (&context
                * BigInt::from(31))
                + &cipher_val_bigint
                + &position_factor;

            context = context
                % BigInt::from(
                    0xFFFFFFFF_u64,
                );
        } else {

            log::warn!(
                "Chunk {} could not \
                 be solved",
                i
            );
        }
    }

    if full_message_bytes.is_empty() {

        return Err("No numerical \
                    solutions found \
                    for any chunk. \
                    Is the key \
                    correct?"
            .to_string());
    }

    let message =
        String::from_utf8_lossy(
            &full_message_bytes,
        )
        .to_string();

    Ok(message)
}

#[server(DecryptSurprise, "/api")]

pub async fn decrypt_surprise_server(
    key: String
) -> Result<String, ServerFnError> {

    #[cfg(feature = "ssr")]
    {

        use crate::MODEL;

        let raw_message =
            decrypt_surprise_logic(
                &key,
            )
            .await
            .map_err(|e| {

                ServerFnError::new(e)
            })?;

        log::info!(
            "Surprise decrypted: {}",
            raw_message
        );

        // Attempt AI interpretation
        let model_arc = {

            let model_lock =
                MODEL.lock().await;

            model_lock.clone()
        };

        if let Some(m) = model_arc {

            let prompt = format!(
                "<|user|>\nYou are a \
                 helpful and creative \
                 assistant. I have \
                 decrypted a hidden \
                 message from a \
                 secret surprise box. \
                 The message is: \
                 \"{}\". Please \
                 explain what this \
                 means or give a \
                 funny, creative \
                 interpretation of it \
                 in one or two \
                 sentences. Keep it \
                 surprising and \
                 delightful!<|end|>\\
                 n<|assistant|>\n",
                raw_message
            );

            let res = tokio::task::spawn_blocking(move || {
                let mut model = m.lock().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
                model.complete(&prompt, 150, || false)
            }).await.map_err(|e| ServerFnError::new(format!("Join error: {}", e)))?;

            match res {
                | Ok(
                    interpretation,
                ) => {
                    Ok(interpretation
                        .trim()
                        .to_string())
                },
                | Err(e) => {

                    log::error!("AI interpretation failed: {}", e);

                    Ok(raw_message)
                },
            }
        } else {

            Ok(raw_message)
        }
    }

    #[cfg(not(feature = "ssr"))]
    {

        let _ = key;

        Err(ServerFnError::new(
            "Only available on SSR",
        ))
    }
}

#[component]

pub fn SurpriseModal(
    show: Signal<bool>,
    on_close: Callback<()>,
) -> impl IntoView {

    let (key_input, set_key_input) =
        signal("".to_string());

    let decrypt_action =
        Action::new(|key: &String| {

            let key = key.clone();

            async move {

                decrypt_surprise_server(
                    key,
                )
                .await
            }
        });

    let result = decrypt_action.value();

    let pending =
        decrypt_action.pending();

    view! {
        <Show when=move || show.get()>
            <div class="fixed inset-0 z-[200] flex items-center justify-center p-6 bg-black/40 backdrop-blur-md animate-fade-in" on:click=move |_| on_close.run(())>
                <div class="glass-dark border border-white/10 rounded-[2.5rem] w-full max-w-lg shadow-[0_50px_100px_-20px_rgba(0,0,0,0.5)] overflow-hidden animate-scale-in" on:click=|ev| ev.stop_propagation()>
                    <div class="px-8 py-6 border-b border-white/5 flex justify-between items-center">
                        <div class="flex items-center gap-3">
                            <div class="p-2 bg-obsidian-accent/10 rounded-xl">
                                <svg class="w-5 h-5 text-obsidian-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                </svg>
                            </div>
                            <h2 class="text-xl font-black text-obsidian-heading tracking-tight">"Cipher Box"</h2>
                        </div>
                        <button on:click=move |_| on_close.run(()) class="w-10 h-10 rounded-full flex items-center justify-center text-obsidian-text/40 hover:bg-white/5 hover:text-white transition-all">"✕"</button>
                    </div>

                    <div class="p-8 space-y-8 text-center">
                        <div class="relative py-4 flex justify-center">
                            <div class="absolute inset-0 bg-obsidian-accent/20 blur-3xl rounded-full scale-50"></div>
                            <div class="relative w-20 h-20 bg-gradient-to-br from-obsidian-accent to-obsidian-accent-light rounded-3xl flex items-center justify-center shadow-2xl shadow-obsidian-accent/30 animate-pulse">
                                <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                </svg>
                            </div>
                        </div>

                        <div class="space-y-4">
                            <p class="text-[12px] text-obsidian-text/40 font-bold uppercase tracking-[0.2em]">"Mathematical Key Injection"</p>
                            <input
                                type="text"
                                placeholder="Input expr (e.g. x^6)"
                                class="w-full bg-black/40 border border-white/5 rounded-2xl px-6 py-4 text-white focus:ring-2 focus:ring-obsidian-accent/30 outline-none transition-all placeholder:text-white/5 text-center font-mono text-lg shadow-inner"
                                on:input=move |ev| set_key_input.set(event_target_value(&ev))
                                prop:value=key_input
                                on:keydown=move |ev| {
                                    if ev.key() == "Enter" {
                                        let _ = decrypt_action.dispatch(key_input.get());
                                    }
                                }
                            />
                            <button
                                on:click=move |_| {let _ = decrypt_action.dispatch(key_input.get());}
                                class="w-full bg-gradient-to-br from-obsidian-accent to-obsidian-accent/80 hover:brightness-110 text-white font-black uppercase tracking-[0.2em] text-[11px] py-4 rounded-2xl transition-all shadow-[0_10px_30px_-10px_rgba(59,130,246,0.6)] disabled:opacity-30 active:scale-95 group"
                                disabled=pending
                            >
                                <Show when=move || pending.get() fallback=|| "Initiate Decryption">
                                    <div class="flex items-center justify-center gap-2">
                                        <div class="w-3 h-3 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
                                        "Cracking..."
                                    </div>
                                </Show>
                            </button>
                        </div>

                        <Transition fallback=|| ()>
                            {move || result.get().map(|res| match res {
                                Ok(msg) => view! {
                                    <div class="mt-8 p-8 bg-white/2 border border-obsidian-accent/20 rounded-3xl animate-scale-in relative group overflow-hidden">
                                        <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-obsidian-accent to-transparent opacity-50"></div>
                                        <div class="text-[9px] font-black uppercase tracking-[0.3em] text-obsidian-accent/40 mb-4">"Interpreted Signal"</div>
                                        <p class="text-white leading-relaxed italic text-lg font-bold tracking-tight">"\"" {msg} "\""</p>
                                        <div class="mt-6 flex justify-center gap-1.5 opacity-20">
                                            <div class="w-1 h-1 bg-obsidian-accent rounded-full animate-bounce"></div>
                                            <div class="w-1 h-1 bg-obsidian-accent rounded-full animate-bounce delay-150"></div>
                                            <div class="w-1 h-1 bg-obsidian-accent rounded-full animate-bounce delay-300"></div>
                                        </div>
                                    </div>
                                }.into_any(),
                                Err(e) => view! {
                                    <div class="mt-8 p-4 bg-red-500/5 border border-red-500/10 rounded-2xl animate-shake">
                                        <p class="text-red-400 text-[11px] font-black uppercase tracking-wider">{e.to_string()}</p>
                                    </div>
                                }.into_any(),
                            })}
                        </Transition>
                    </div>
                </div>
            </div>
        </Show>
    }
}

#[cfg(feature = "ssr")]

// Helper for fuzzy comparison of float constants in expressions
fn is_fuzzy_equal(
    a: &Expr,
    b: &Expr,
) -> bool {

    use num_traits::ToPrimitive;

    // Collect leaf terms to handle structural differences (Add vs AddList, ordering, etc.)
    fn collect_terms(
        e: &Expr,
        big_acc: &mut BigInt,
        float_acc: &mut f64,
        sign: i32,
    ) -> bool {

        match e {
            | Expr::Add(l, r) => {

                let a = collect_terms(
                    l,
                    big_acc,
                    float_acc,
                    sign,
                );

                let b = collect_terms(
                    r,
                    big_acc,
                    float_acc,
                    sign,
                );

                a || b
            },
            | Expr::Sub(l, r) => {

                let a = collect_terms(
                    l,
                    big_acc,
                    float_acc,
                    sign,
                );

                let b = collect_terms(
                    r,
                    big_acc,
                    float_acc,
                    -sign,
                );

                a || b
            },
            | Expr::AddList(list) => {

                let mut shifted = false;

                for item in list {

                    if collect_terms(
                        item,
                        big_acc,
                        float_acc,
                        sign,
                    ) {

                        shifted = true;
                    }
                }

                shifted
            },
            | Expr::Neg(inner) => {
                collect_terms(
                    inner,
                    big_acc,
                    float_acc,
                    -sign,
                )
            },
            | Expr::BigInt(n) => {

                if sign > 0 {

                    *big_acc += n;
                } else {

                    *big_acc -= n;
                }

                true
            },
            | Expr::Constant(f) => {

                if sign > 0 {

                    *float_acc += f;
                } else {

                    *float_acc -= f;
                }

                true
            },
            | Expr::Mul(_l, _r) => {

                // If it's a Mul, we can't easily flatten, but we can try to eval it
                // This is needed if force_bigint failed to fold it.
                if let Some(f) =
                    try_eval_f(e)
                {

                    if sign > 0 {

                        *float_acc += f;
                    } else {

                        *float_acc -= f;
                    }

                    true
                } else {

                    false
                }
            },
            | _ => {

                if let Some(f) =
                    try_eval_f(e)
                {

                    if sign > 0 {

                        *float_acc += f;
                    } else {

                        *float_acc -= f;
                    }

                    true
                } else {

                    false
                }
            },
        }
    }

    fn try_eval_f(
        e: &Expr
    ) -> Option<f64> {

        match e {
            | Expr::BigInt(b) => {
                b.to_f64()
            },
            | Expr::Constant(f) => {
                Some(*f)
            },
            | Expr::Mul(l, r) => {
                try_eval_f(l).and_then(
                    |la| {
                        try_eval_f(r)
                            .map(|lb| {
                                la * lb
                            })
                    },
                )
            },
            | Expr::Power(l, r) => {
                try_eval_f(l).and_then(
                    |la| {
                        try_eval_f(r)
                            .map(|lb| {
                                la.powf(
                                    lb,
                                )
                            })
                    },
                )
            },
            | Expr::Div(l, r) => {
                try_eval_f(l).and_then(
                    |la| {
                        try_eval_f(r)
                            .map(|lb| {
                                la / lb
                            })
                    },
                )
            },
            | Expr::Sin(a) => {
                try_eval_f(a)
                    .map(|v| v.sin())
            },
            | Expr::Cos(a) => {
                try_eval_f(a)
                    .map(|v| v.cos())
            },
            | Expr::Abs(a) => {
                try_eval_f(a)
                    .map(|v| v.abs())
            },
            | Expr::Sqrt(a) => {
                try_eval_f(a)
                    .map(|v| v.sqrt())
            },
            | Expr::AddList(list) => {

                let mut sum = 0.0;

                for item in list {

                    sum += try_eval_f(
                        item,
                    )?;
                }

                Some(sum)
            },
            | _ => None,
        }
    }

    let mut big_a = BigInt::from(0);

    let mut float_a = 0.0;

    let mut big_b = BigInt::from(0);

    let mut float_b = 0.0;

    let has_a = collect_terms(
        a,
        &mut big_a,
        &mut float_a,
        1,
    );

    let has_b = collect_terms(
        b,
        &mut big_b,
        &mut float_b,
        1,
    );

    if !has_a || !has_b {

        return a == b; // Fallback to strict equality if we couldn't even collect terms
    }

    // Compare BigInt parts
    if big_a != big_b {

        // If they differ, check if the difference can be absorbed by the float part
        let diff = &big_a - &big_b;

        if let Some(d_f) = diff.to_f64()
        {

            float_a += d_f;
        } else {

            return false;
        }
    }

    // Compare float parts with robust tolerance
    let diff =
        (float_a - float_b).abs();

    if float_a.abs() < 1.0
        && float_b.abs() < 1.0
    {

        diff < 1e-4
    } else {

        let max_val = float_a
            .abs()
            .max(float_b.abs());

        diff / max_val < 1e-11
    }
}

#[cfg(feature = "ssr")]

fn try_solve_bigint(
    deriv: &Expr,
    target: &Expr,
    offset: &BigInt,
) -> Option<BigInt> {

    use num_bigint::BigInt;
    use num_traits::Signed;
    use rssn::symbolic::calculus::substitute;
    use rssn::symbolic::simplify_dag::simplify;

    log::info!(
        "try_solve_bigint \
         (brute-force strategy) start"
    );

    // Pre-simplify the derivative to speed up substitution in the loop
    let simplified_deriv =
        simplify(deriv)
            .to_ast()
            .unwrap_or(deriv.clone());

    use rayon::prelude::*;

    // Parallel brute-force search
    // We can search the range [0, 255] in parallel
    let result = (0 ..= 255)
        .into_par_iter()
        .find_map_any(|val| {

            let x_guess =
                BigInt::from(256 + val)
                    + offset; // range [256+offset, 511+offset]

            let eval_expr = substitute(
                &simplified_deriv,
                "x",
                &Expr::BigInt(
                    x_guess.clone(),
                ),
            );

            let eval_res =
                force_bigint(eval_expr);

            if is_fuzzy_equal(
                &eval_res,
                target,
            ) {

                return Some(x_guess);
            }

            None
        });

    if result.is_some() {

        return result;
    }

    log::info!(
        "Brute-force failed to find \
         solution in [0, 255]"
    );

    // Fallback search for larger ranges if needed (e.g. 2-byte chunks)
    let result_large = (0 ..= 65535)
        .into_par_iter()
        .find_map_any(|val| {

            let x_guess = BigInt::from(
                65536 + val,
            ) + offset;

            let eval_expr = substitute(
                &simplified_deriv,
                "x",
                &Expr::BigInt(
                    x_guess.clone(),
                ),
            );

            let eval_res =
                force_bigint(eval_expr);

            if is_fuzzy_equal(
                &eval_res,
                target,
            ) {

                return Some(x_guess);
            }

            None
        });

    if result_large.is_some() {

        return result_large;
    }

    log::info!(
        "Brute-force failed to find \
         solution in [256, 131071]"
    );

    None
}


#[cfg(feature = "ssr")]

fn force_bigint(
    root_expr: Expr
) -> Expr {

    use std::sync::Arc;

    use num_traits::Signed;
    use num_traits::ToPrimitive;
    use num_traits::Zero;
    use rssn::symbolic::core::Expr;

    // We start by unwrapping the AST if needed
    let root = root_expr
        .to_ast()
        .unwrap_or(root_expr.clone());

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

    // Stack for post-order traversal simulation
    // (Node, visited_children)
    // If visited_children is true, we pop input from output stack and compute
    let mut visit_stack =
        vec![(root, false)];

    let mut output_stack: Vec<Expr> =
        Vec::new();

    while let Some((expr, visited)) =
        visit_stack.pop()
    {

        if visited {

            // Children have been processed and are on output_stack.
            // Reconstruct and optimize.
            match expr {
                Expr::Add(_, _) => {
                    let rhs = output_stack.pop().unwrap();
                    let lhs = output_stack.pop().unwrap();
                    match (lhs, rhs) {
                        (Expr::BigInt(la), Expr::BigInt(lb)) => output_stack.push(Expr::BigInt(la + lb)),
                        (Expr::Constant(la), Expr::Constant(lb)) => output_stack.push(Expr::Constant(la + lb)),
                        (Expr::BigInt(la), Expr::Constant(lb)) => {
                            // If BigInt is very large, keep separate to preserve precision
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
                Expr::Sub(_, _) => {
                    let rhs = output_stack.pop().unwrap();
                    let lhs = output_stack.pop().unwrap();
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
                Expr::Mul(_, _) => {
                    let rhs = output_stack.pop().unwrap();
                    let lhs = output_stack.pop().unwrap();
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
                Expr::Div(_, _) => {
                    let rhs = output_stack.pop().unwrap();
                    let lhs = output_stack.pop().unwrap();
                    match (lhs, rhs) {
                        (Expr::BigInt(la), Expr::BigInt(lb)) if !lb.is_zero() && &la % &lb == BigInt::from(0) => output_stack.push(Expr::BigInt(la / lb)),
                        (Expr::Constant(la), Expr::Constant(lb)) => output_stack.push(Expr::Constant(la / lb)),
                        (Expr::BigInt(la), Expr::Constant(lb)) => output_stack.push(Expr::Constant(la.to_f64().unwrap_or(0.0) / lb)),
                        (Expr::Constant(la), Expr::BigInt(lb)) => output_stack.push(Expr::Constant(la / lb.to_f64().unwrap_or(0.0))), // Handle div by zero? Rust f64 handles inf.
                        (l, r) => output_stack.push(Expr::Div(Arc::new(l), Arc::new(r))),
                    }
                },
                Expr::Power(_, _) => {
                    let rhs = output_stack.pop().unwrap();
                    let lhs = output_stack.pop().unwrap();
                    match (lhs, rhs) {
                        (Expr::BigInt(base), Expr::BigInt(exp)) => {
                            if let Some(e) = exp.to_u32() {
                                output_stack.push(Expr::BigInt(base.pow(e)));
                            } else if exp == BigInt::from(0) {
                                output_stack.push(Expr::BigInt(BigInt::from(1)));
                            } else {
                                // Fallback to float if exponential is too large?
                                // Or symbolic.
                                // Let's keep existing logic for pure BigInt, but add float fallback
                                output_stack.push(Expr::Power(Arc::new(Expr::BigInt(base)), Arc::new(Expr::BigInt(exp))));
                            }
                        }
                        (Expr::Constant(b), Expr::Constant(e)) => output_stack.push(Expr::Constant(b.powf(e))),
                        (Expr::BigInt(b), Expr::Constant(e)) => output_stack.push(Expr::Constant(b.to_f64().unwrap_or(0.0).powf(e))),
                        (Expr::Constant(b), Expr::BigInt(e)) => output_stack.push(Expr::Constant(b.powf(e.to_f64().unwrap_or(0.0)))),
                        (l, r) => output_stack.push(Expr::Power(Arc::new(l), Arc::new(r))),
                    }
                },
                Expr::LogBase(_, _) => {
                    let rhs = output_stack.pop().unwrap();
                    let lhs = output_stack.pop().unwrap();
                    output_stack.push(Expr::LogBase(Arc::new(lhs), Arc::new(rhs)));
                },
                Expr::Neg(_) => {
                    let inner = output_stack.pop().unwrap();
                    match inner {
                        Expr::BigInt(n) => output_stack.push(Expr::BigInt(-n)),
                        other => output_stack.push(Expr::Neg(Arc::new(other))),
                    }
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
                    let len = list.len();
                    // Items are stuck on output stack in reverse order of processing?
                    // We pushed children in reverse order so they were popped in order.
                    // So key is: push children C, B, A  -> pops A, B, C.
                    // Wait, we push onto visit_stack. Last pushed is popped first.
                    // If we push A, then B. B is processed first. Result B is on stack. Then A. Result A on stack.
                    // So output stack has [.., Result B, Result A].
                    // We need to reverse taking them off.

                    let mut new_list = Vec::with_capacity(len);
                    for _ in 0..len {
                        new_list.push(output_stack.pop().unwrap());
                    }
                    new_list.reverse();
                    output_stack.push(Expr::AddList(new_list));
                },
                Expr::MulList(list) => {
                    let len = list.len();
                    let mut new_list = Vec::with_capacity(len);
                    for _ in 0..len {
                        new_list.push(output_stack.pop().unwrap());
                    }
                    new_list.reverse();
                    output_stack.push(Expr::MulList(new_list));
                },
                // Leaves already handled when !visited
                _ => unreachable!("Should not visit leaves in post-process"),
            }
        } else {

            // First time seeing this node.
            // Check if it's a leaf that we can process immediately
            let processed_leaf = match &expr {
                Expr::Constant(f) if f.fract() == 0.0 => Some(Expr::BigInt(num_bigint::BigInt::from(*f as i64))),
                Expr::Rational(r) if r.is_integer() => Some(Expr::BigInt(r.to_integer())),
                // Other leaves pass through
                Expr::Constant(_) | Expr::Rational(_) | Expr::Variable(_) | Expr::BigInt(_) => Some(expr.clone()),
                _ => None,
            };

            if let Some(leaf) =
                processed_leaf
            {

                output_stack.push(leaf);

                continue;
            }

            // It's a non-leaf node. Push back with visited=true, then children.
            // We want to process Left then Right.
            // Stack is LIFO. So push Right then Left.
            // Then Left is popped first, processed, output on stack.
            // Then Right is popped, processed, output on stack.
            // So output_stack will have [L_res, R_res].

            // However, we need to ownership of inner Arcs.
            // The expr we have is the one we will match on later.
            // clone children

            // Fix borrow checker errors by using ref and avoiding partial moves

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

                    let b = b.as_ref();

                    let a = a.as_ref();

                    let b_expr = b
                        .clone()
                        .to_ast()
                        .unwrap_or(
                            b.clone(),
                        );

                    let a_expr = a
                        .clone()
                        .to_ast()
                        .unwrap_or(
                            a.clone(),
                        );

                    visit_stack.push((
                        b_expr,
                        false,
                    ));

                    visit_stack.push((
                        a_expr,
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

                    let a = a.as_ref();

                    let a_expr = a
                        .clone()
                        .to_ast()
                        .unwrap_or(
                            a.clone(),
                        );

                    visit_stack.push((
                        a_expr,
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

                    // We want results [0, 1, 2].
                    // If we push 2, 1, 0.
                    // 0 pops, Res0. 1 pops, Res1. 2 pops, Res2.
                    // Output: [.., Res0, Res1, Res2].
                    // Then we can pop them in reverse to build vector.

                    // So we push in reverse order (e.g. 5, 4... 0).
                    // Iterator is 0..N. .rev() gives N-1..0.
                    for item in list
                        .iter()
                        .rev()
                    {

                        let it = item.clone().to_ast().unwrap_or(item.clone());

                        visit_stack
                            .push((
                                it,
                                false,
                            ));
                    }
                },
                | _ => {

                    unreachable!(
                    "Leaves handled \
                     above"
                )
                },
            }
        }
    }

    output_stack
        .pop()
        .unwrap_or(root_expr) // Should be exactly one item if logic holds
}
