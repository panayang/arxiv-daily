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

use clap::Parser;

const LONG_VERSION: &str = concat!(
    "\narXiv Daily Dashboard Build Information\n",
    "==================================================\n",
    "SOFTWARE INFO\n",
    "  Version:          ", env!("CARGO_PKG_VERSION"), "\n",
    "  Authors:          ", env!("CARGO_PKG_AUTHORS"), "\n",
    "  Repository:       ", env!("VERGEN_GIT_DESCRIBE"), "\n",
    "\n",
    "BUILD METADATA\n",
    "  Built At:         ", env!("VERGEN_BUILD_TIMESTAMP"), "\n",
    "  Optimization:     Level ", env!("VERGEN_CARGO_OPT_LEVEL"), "\n",
    "  Debug Symbols:    ", env!("VERGEN_CARGO_DEBUG"), "\n",
    "  Target Triple:    ", env!("VERGEN_CARGO_TARGET_TRIPLE"), "\n",
    "\n",
    "GIT TELEMETRY\n",
    "  Branch:           ", env!("VERGEN_GIT_BRANCH"), "\n",
    "  Commit SHA:       ", env!("VERGEN_GIT_SHA"), "\n",
    "  Author:           ", env!("VERGEN_GIT_COMMIT_AUTHOR_NAME"), " <", env!("VERGEN_GIT_COMMIT_AUTHOR_EMAIL"), ">\n",
    "  Commit Msg:       ", env!("VERGEN_GIT_COMMIT_MESSAGE"), "\n",
    "\n",
    "COMPILER INFO\n",
    "  Rustc Version:    ", env!("VERGEN_RUSTC_SEMVER"), "\n",
    "  LLVM Version:     ", env!("VERGEN_RUSTC_LLVM_VERSION"), "\n",
    "  Host Triple:      ", env!("VERGEN_RUSTC_HOST_TRIPLE"), "\n",
    "\n",
    "BUILD HOST SPECS\n",
    "  OS:               ", env!("VERGEN_SYSINFO_NAME"), " (", env!("VERGEN_SYSINFO_OS_VERSION"), ")\n",
    "  Kernel:           ", env!("VERGEN_SYSINFO_KERNEL_VERSION"), "\n",
    "  CPU:              ", env!("VERGEN_SYSINFO_CPU_BRAND"), "\n",
    "  Cores:            ", env!("VERGEN_SYSINFO_CPU_CORE_COUNT"), "\n",
    "  Total Memory:     ", env!("VERGEN_SYSINFO_TOTAL_MEMORY"), "\n",
    "=================================================="
);

#[derive(Parser)]
#[command(
    name = "arXiv Daily Dashboard",
    author,
    version,
    long_version = LONG_VERSION,
    about = "This is arXiv-daily: A dashboard for daily arXiv papers"
)]

struct Args {
    #[arg(short, long)]
    name: Option<String>,
}

#[cfg(feature = "ssr")]
#[tokio::main]

async fn main() {

    let default_level =
        if cfg!(debug_assertions) {

            "info"
        } else {

            "error"
        };

    if std::env::var("RUST_LOG")
        .is_err()
    {

        #[allow(unsafe_code)]
        unsafe {

            std::env::set_var(
                "RUST_LOG",
                default_level,
            );
        }
    }

    env_logger::init();

    // Configure rustls to use aws-lc-rs
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("Failed to install default crypto provider");

    let _args = Args::parse();

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
        .layer(tower_http::set_header::SetResponseHeaderLayer::overriding(
            http::header::ALT_SVC,
            http::header::HeaderValue::from_str(&format!("h3=\":{}\"; ma=86400", addr.port())).unwrap(),
        ))
        .with_state(leptos_options);

    let tls_cert = include_bytes!(
        "../assets/cert.pem"
    );

    let tls_key = include_bytes!(
        "../assets/key.pem"
    );

    println!(
        "listening on https://{}",
        &addr
    );

    #[cfg(not(debug_assertions))]
    {

        let url =
            format!("https://{}", addr);

        if let Err(e) = open::that(&url)
        {

            eprintln!(
                "Failed to open \
                 browser: {}",
                e
            );
        }
    }

    // Serving logic
    let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem(tls_cert.to_vec(), tls_key.to_vec()).await.unwrap();

    let h1_h2_server =
        axum_server::bind_rustls(
            addr,
            tls_config,
        )
        .serve(
            app.clone()
                .into_make_service(),
        );

    // HTTP/3 (QUIC) server
    let h3_server = async {

        // Load keys for Quinn (needs rustls::ServerConfig)
        let certs = rustls_pemfile::certs(&mut &tls_cert[..])
            .collect::<Result<Vec<_>, _>>()
            .expect("Valid certs");

        let key = rustls_pemfile::private_key(&mut &tls_key[..])
            .expect("Valid key")
            .expect("Valid key found");

        let mut server_crypto = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .expect("Valid config");

        server_crypto.alpn_protocols =
            vec![b"h3".to_vec()];

        let server_config = quinn::ServerConfig::with_crypto(std::sync::Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto).unwrap()
        ));

        let endpoint =
            quinn::Endpoint::server(
                server_config,
                addr,
            )
            .unwrap();

        while let Some(new_conn) =
            endpoint
                .accept()
                .await
        {

            let app = app.clone();

            tokio::spawn(async move {

                match new_conn.await {
                    | Ok(conn) => {

                        let mut h3_conn = h3::server::Connection::new(h3_quinn::Connection::new(conn)).await.unwrap();

                        let app =
                            app.clone();

                        tokio::spawn(
                            async move {

                                loop {

                                    match h3_conn.accept().await {
                                    Ok(Some(resolver)) => {
                                        let app = app.clone();
                                        tokio::spawn(async move {
                                            let (req, stream) = match resolver.resolve_request().await {
                                                 Ok(v) => v,
                                                 Err(e) => {
                                                     eprintln!("Error resolving request: {}", e);
                                                     return;
                                                 }
                                            };
                                            let (mut send_stream, mut recv_stream) = stream.split();

                                            use axum::body::Body;
                                            #[allow(unused_imports)]
                                            use h3::server::RequestStream;
                                            use futures::StreamExt;
                                            use bytes::Bytes;
                                            #[allow(unused_imports)]
                                            use http_body_util::BodyExt;
                                            use async_stream::stream;

                                            let (parts, _) = req.into_parts();
                                            // Convert H3 stream to Axum body
                                            use bytes::Buf; // Import Buf trait
                                            let body_stream = stream! {
                                                while let Ok(Some(mut chunk)) = recv_stream.recv_data().await {
                                                    let bytes = chunk.copy_to_bytes(chunk.remaining());
                                                    yield Ok::<Bytes, axum::Error>(bytes);
                                                }
                                            };
                                            let req_body = Body::from_stream(body_stream);
                                            let axum_req = http::Request::from_parts(parts, req_body);

                                            use tower::ServiceExt;
                                            let service = app.clone();

                                            match service.oneshot(axum_req).await {
                                                Ok(response) => {
                                                    let (parts, body) = response.into_parts();
                                                    let h3_resp = http::Response::from_parts(parts, ());

                                                    if let Err(e) = send_stream.send_response(h3_resp).await {
                                                         eprintln!("Error sending response head: {}", e);
                                                         return;
                                                    }

                                                    // Stream the response body
                                                    let mut body_stream = body.into_data_stream();
                                                    while let Some(chunk) = body_stream.next().await {
                                                        if let Ok(bytes) = chunk {
                                                            if let Err(e) = send_stream.send_data(bytes).await {
                                                                 eprintln!("Error sending response data: {}", e);
                                                                 break;
                                                            }
                                                        }
                                                    }
                                                    let _ = send_stream.finish().await;
                                                }
                                                Err(e) => eprintln!("Service error: {}", e),
                                            }
                                        });
                                    }
                                    Ok(None) => break, // Connection closed
                                    Err(e) => {
                                        eprintln!("H3 accept error: {}", e);
                                        break;
                                    }
                                }
                                }
                            },
                        );
                    },
                    | Err(e) => {

                        eprintln!("Error accepting connection: {}", e);
                    },
                }
            });
        }
    };

    tokio::select! {
        _ = h1_h2_server => {},
        _ = h3_server => {},
    }
}

#[cfg(not(feature = "ssr"))]

pub fn main() {
    // no-op for non-ssr builds
}
