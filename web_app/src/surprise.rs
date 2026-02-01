#![allow(unused_imports)]

use leptos::prelude::*;
use num_bigint::BigInt;
#[cfg(feature = "ssr")]
use rssn::symbolic::core::Expr;

#[cfg(feature = "ssr")]

pub async fn decrypt_surprise_logic(
    user_key: &str
) -> Result<String, String> {

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

    for (i, encrypted_expr) in
        encrypted_exprs
            .into_iter()
            .enumerate()
    {

        let mut real_roots = Vec::new();

        // Strategy 1: BigInt-aware solver (exact for large integers, handles up to quadratic)
        if let Some(root) =
            try_solve_bigint(
                &deriv,
                &encrypted_expr,
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

        if let Some(target_num) =
            real_roots.first()
        {

            let mut bytes = target_num
                .to_bytes_be()
                .1;

            if !bytes.is_empty() {

                bytes.remove(0); // Remove sentinel byte
                full_message_bytes
                    .extend(bytes);
            }
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
                                on:click=move |_| { let _ = decrypt_action.dispatch(key_input.get()); }
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

fn try_solve_bigint(
    deriv: &Expr,
    target: &Expr,
) -> Option<BigInt> {

    use num_bigint::BigInt;
    use rssn::symbolic::calculus::substitute;
    use rssn::symbolic::simplify_dag::simplify;

    log::info!(
        "try_solve_bigint \
         (brute-force strategy) start"
    );

    let target_val = match force_bigint(
        target.clone(),
    ) {
        | Expr::BigInt(n) => n,
        | _ => return None,
    };

    // Since we now use 1-byte chunks with a 0x01 sentinel,
    // x will always be in the range [256, 511].
    // This brute-force approach is foolproof for any complex key.
    for val in 0 ..= 255 {

        let x_guess =
            BigInt::from(256 + val);

        let eval =
            simplify(&substitute(
                deriv,
                "x",
                &Expr::BigInt(
                    x_guess.clone(),
                ),
            ));

        if let Expr::BigInt(res) =
            force_bigint(eval)
        {

            if res == target_val {

                return Some(x_guess);
            }
        }
    }

    // Fallback search for larger ranges if needed (e.g. 2-byte chunks)
    for val in 0 ..= 65535 {

        let x_guess =
            BigInt::from(65536 + val);

        let eval =
            simplify(&substitute(
                deriv,
                "x",
                &Expr::BigInt(
                    x_guess.clone(),
                ),
            ));

        if let Expr::BigInt(res) =
            force_bigint(eval)
        {

            if res == target_val {

                return Some(x_guess);
            }
        }
    }

    log::warn!(
        "Brute-force failed to find \
         solution in [256, 131071]"
    );

    None
}


#[cfg(feature = "ssr")]

fn force_bigint(expr: Expr) -> Expr {

    use std::sync::Arc;

    use num_traits::ToPrimitive;
    use rssn::symbolic::core::Expr;

    let expr = expr
        .to_ast()
        .unwrap_or(expr);

    match expr {
        Expr::Constant(f) if f.fract() == 0.0 => Expr::BigInt(num_bigint::BigInt::from(f as i64)),
        Expr::Rational(r) if r.is_integer() => Expr::BigInt(r.to_integer()),
        Expr::Add(a, b) => {
            let lhs = force_bigint(a.as_ref().clone());
            let rhs = force_bigint(b.as_ref().clone());
            match (lhs, rhs) {
                (Expr::BigInt(la), Expr::BigInt(lb)) => Expr::BigInt(la + lb),
                (l, r) => Expr::Add(Arc::new(l), Arc::new(r)),
            }
        },
        Expr::Sub(a, b) => {
            let lhs = force_bigint(a.as_ref().clone());
            let rhs = force_bigint(b.as_ref().clone());
            match (lhs, rhs) {
                (Expr::BigInt(la), Expr::BigInt(lb)) => Expr::BigInt(la - lb),
                (l, r) => Expr::Sub(Arc::new(l), Arc::new(r)),
            }
        },
        Expr::Mul(a, b) => {
            let lhs = force_bigint(a.as_ref().clone());
            let rhs = force_bigint(b.as_ref().clone());
            match (lhs, rhs) {
                (Expr::BigInt(la), Expr::BigInt(lb)) => Expr::BigInt(la * lb),
                (l, r) => Expr::Mul(Arc::new(l), Arc::new(r)),
            }
        },
        Expr::Div(a, b) => {
            let lhs = force_bigint(a.as_ref().clone());
            let rhs = force_bigint(b.as_ref().clone());
            match (lhs, rhs) {
                (Expr::BigInt(la), Expr::BigInt(lb)) if &la % &lb == BigInt::from(0) => Expr::BigInt(la / lb),
                (l, r) => Expr::Div(Arc::new(l), Arc::new(r)),
            }
        },
        Expr::Power(a, b) => {
            let lhs = force_bigint(a.as_ref().clone());
            let rhs = force_bigint(b.as_ref().clone());
            match (lhs, rhs) {
                (Expr::BigInt(base), Expr::BigInt(exp)) => {
                    if let Some(e) = exp.to_u32() {
                        Expr::BigInt(base.pow(e))
                    } else if exp == BigInt::from(0) {
                        Expr::BigInt(BigInt::from(1))
                    } else {
                        Expr::Power(Arc::new(Expr::BigInt(base)), Arc::new(Expr::BigInt(exp)))
                    }
                }
                (l, r) => Expr::Power(Arc::new(l), Arc::new(r)),
            }
        },
        Expr::Neg(a) => {
            let inner = force_bigint(a.as_ref().clone());
            match inner {
                Expr::BigInt(n) => Expr::BigInt(-n),
                other => Expr::Neg(Arc::new(other)),
            }
        },
        Expr::AddList(list) => Expr::AddList(list.into_iter().map(force_bigint).collect()),
        Expr::MulList(list) => Expr::MulList(list.into_iter().map(force_bigint).collect()),
        _ => expr,
    }
}
