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
            <div class="fixed inset-0 bg-black/80 backdrop-blur-md z-[200] flex items-center justify-center p-4 animate-in fade-in duration-300">
                <div class="bg-obsidian-sidebar border border-white/10 w-full max-w-lg rounded-3xl shadow-2xl relative overflow-hidden animate-in zoom-in-95 duration-300">
                    <div class="p-8 space-y-6 text-center">
                        <div class="flex items-center justify-between text-left">
                            <h2 class="text-2xl font-black text-white tracking-tight">"Surprise Box"</h2>
                            <button on:click=move |_| on_close.run(()) class="text-white/40 hover:text-white transition-colors text-2xl">"✕"</button>
                        </div>

                        <div class="py-4 flex justify-center">
                            <div class="w-16 h-16 bg-gradient-to-br from-obsidian-accent to-obsidian-accent/50 rounded-2xl flex items-center justify-center shadow-lg shadow-obsidian-accent/20 animate-bounce">
                                <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                </svg>
                            </div>
                        </div>

                        <div class="space-y-4">
                            <p class="text-obsidian-text/60 text-sm">"Enter your mathematical key to unlock the secret."</p>
                            <input
                                type="text"
                                placeholder="e.g. x^6"
                                class="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:ring-2 focus:ring-obsidian-accent/50 outline-none transition-all placeholder:text-white/10 text-center font-mono"
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
                                class="w-full bg-obsidian-accent hover:bg-obsidian-accent/80 text-white font-bold py-4 rounded-xl transition-all shadow-lg shadow-obsidian-accent/20 disabled:opacity-50 active:scale-95"
                                disabled=pending
                            >
                                <Show when=move || pending.get() fallback=|| "Unlock Surprise">"Unlocking..."</Show>
                            </button>
                        </div>

                        <Transition fallback=|| ()>
                            {move || result.get().map(|res| match res {
                                Ok(msg) => view! {
                                    <div class="mt-6 p-8 bg-gradient-to-br from-obsidian-accent/10 to-transparent border border-obsidian-accent/20 rounded-2xl animate-in slide-in-from-bottom-8 zoom-in-90 duration-700 relative group">
                                        <div class="absolute -top-3 left-1/2 -translate-x-1/2 bg-obsidian-accent text-[10px] font-black uppercase tracking-widest px-3 py-1 rounded-full text-white">"Decrypted"</div>
                                        <p class="text-obsidian-accent leading-relaxed italic text-lg font-medium">"\"" {msg} "\""</p>
                                        <div class="mt-4 flex justify-center gap-1">
                                            <div class="w-1 h-1 bg-obsidian-accent rounded-full animate-ping"></div>
                                            <div class="w-1 h-1 bg-obsidian-accent rounded-full animate-ping delay-100"></div>
                                            <div class="w-1 h-1 bg-obsidian-accent rounded-full animate-ping delay-200"></div>
                                        </div>
                                    </div>
                                }.into_any(),
                                Err(e) => view! {
                                    <div class="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl animate-shake duration-300">
                                        <p class="text-red-400 text-sm font-bold">{e.to_string()}</p>
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
