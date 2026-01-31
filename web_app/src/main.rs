//! This is the arXiv-daily project.

#![recursion_limit = "4096"]
// =========================================================================
// RUST LINT CONFIGURATION: arxiv-daily
// =========================================================================

// -------------------------------------------------------------------------
// LEVEL 1: CRITICAL ERRORS (Deny)
// -------------------------------------------------------------------------
#![deny(
    // Rust Compiler Errors
    dead_code,
    unreachable_code,
    improper_ctypes_definitions,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms,
    clippy::perf,
    clippy::correctness,
    clippy::suspicious,
    clippy::unwrap_used,
    clippy::missing_safety_doc,
    clippy::same_item_push,
    clippy::implicit_clone,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::single_call_fn,
    unsafe_code,
)]
// -------------------------------------------------------------------------
// LEVEL 2: STYLE WARNINGS (Warn)
// -------------------------------------------------------------------------
#![warn(
    missing_docs,
    warnings,
    // To avoid performance issues in hot paths
    clippy::expect_used,
    // To avoid simd optimization issues
    clippy::indexing_slicing,
    // To avoid simd optimization issues
    clippy::arithmetic_side_effects,
    // Possible Truncation warnned, due to CPU branch prediction and simd optimization programs, we will just warn this problems instead deny it.
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::dbg_macro,
    clippy::todo,
    // This is usually a sign of dead code --- but for development purposes, we will just warn it.
    clippy::used_underscore_binding,
    clippy::unnecessary_safety_comment,
    // We thinks do not collapsible if makes the code more extensible.
    clippy::collapsible_if,
    clippy::collapsible_match,
    clippy::collapsible_else_if,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::redundant_else,
    clippy::needless_continue,
    clippy::manual_let_else,
    clippy::no_effect_underscore_binding,
    clippy::must_use_candidate
)]
// -------------------------------------------------------------------------
// LEVEL 3: ALLOW/IGNORABLE (Allow)
// -------------------------------------------------------------------------
#![allow(
    clippy::restriction,
    clippy::inline_always,
    unused_doc_comments,
    clippy::empty_line_after_doc_comments,
    clippy::empty_line_after_outer_attr,
    // It is always reporting on normal math writings.
    clippy::doc_markdown
)]

#[cfg(feature = "ssr")]
#[tokio::main]

async fn main() {

    let args: Vec<String> =
        std::env::args().collect();

    if args.contains(
        &"--version".to_string(),
    ) || args
        .contains(&"-v".to_string())
    {

        println!(
            "arXiv Daily Dashboard"
        );

        println!(
            "Version: {}",
            env!("VERGEN_BUILD_SEMVER")
        );

        println!("Build Timestamp: {}", env!("VERGEN_BUILD_TIMESTAMP"));

        println!(
            "Git SHA: {}",
            env!("VERGEN_GIT_SHA")
        );

        println!(
            "Rustc Version: {}",
            env!("VERGEN_RUSTC_SEMVER")
        );

        return;
    }


    use axum::Router;
    use axum::routing::post;
    use leptos::prelude::*;
    use leptos_axum::LeptosRoutes;
    use leptos_axum::generate_route_list;
    use serde::Deserialize;
    use web_app::App;
    use web_app::shell;

    #[derive(Deserialize)]

    struct ServerConfig {
        ip: String,
        port: u16,
        site_root: String,
    }

    #[derive(Deserialize)]

    struct Config {
        server: Option<ServerConfig>,
    }

    let mut conf =
        get_configuration(None)
            .unwrap();

    // Attempt to load override config
    if let Ok(content) =
        std::fs::read_to_string(
            "config.toml",
        )
        && let Ok(config) =
            toml::from_str::<Config>(
                &content,
            )
        && let Some(server) =
            config.server
    {

        let addr_str = format!(
            "{}:{}",
            server.ip, server.port
        );

        conf.leptos_options
            .site_addr = addr_str
            .parse()
            .unwrap_or(
                conf.leptos_options
                    .site_addr,
            );

        // Only override site_root if LEPTOS_SITE_ROOT is NOT set (i.e., not running via cargo-leptos)
        if std::env::var(
            "LEPTOS_SITE_ROOT",
        )
        .is_err()
        {

            conf.leptos_options
                .site_root = server
                .site_root
                .into();
        }
    }

    let leptos_options =
        conf.leptos_options;

    let addr = leptos_options.site_addr;

    let routes =
        generate_route_list(App);

    // build our application with a route
    let app = Router::new()
        .route("/api/{*fn_name}", post(leptos_axum::handle_server_fns))
        .leptos_routes(&leptos_options, routes, {
            let options = leptos_options.clone();
            move || shell(options.clone())
        })
        .fallback(leptos_axum::file_and_error_handler(shell))
        .with_state(leptos_options);

    // run our app with hyper
    let listener =
        tokio::net::TcpListener::bind(
            &addr,
        )
        .await
        .unwrap();

    println!(
        "listening on http://{}",
        &addr
    );

    #[cfg(not(debug_assertions))]
    {

        let url =
            format!("http://{}", addr);

        if let Err(e) = open::that(&url)
        {

            eprintln!(
                "Failed to open \
                 browser: {}",
                e
            );
        }
    }

    axum::serve(listener, app)
        .await
        .unwrap();
}

#[cfg(not(feature = "ssr"))]

pub fn main() {
    // no-op for non-ssr builds
}
