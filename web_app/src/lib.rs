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

//! This is the library for the arxiv-daily project.

#![allow(
    clippy::empty_line_after_outer_attr
)]
#![recursion_limit = "4096"]

#[cfg(feature = "ssr")]
use std::sync::Arc;
use std::sync::LazyLock;

use bincode_next::config;
use bitcode::Decode;
use bitcode::Encode;
use latex2mathml::DisplayStyle;
use latex2mathml::latex_to_mathml;
use leptos::prelude::*;
use leptos::server_fn::codec::Bitcode;
use leptos::server_fn::codec::Streaming;
use leptos::task::spawn_local;
use leptos_meta::*;
use leptos_router::components::*;
use leptos_router::*;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use shared::Category;
use shared::Paper;
#[cfg(feature = "ssr")]
use tokio::sync::Mutex;
#[allow(unused_imports)]
#[cfg(feature = "ssr")]
use tokio::sync::OnceCell;

#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    bincode_next::Encode,
    bincode_next::Decode,
)]

pub enum FilterStreamMessage {
    Result(String, bool), /* (paper_id, keep) */
    Stopped, /* Stream stopped (e.g. model unloaded) */
    Error(String), // Recoverable error
}

#[allow(unused_variables)]
#[server(GetPapers, "/api", input = Bitcode, output = Bitcode)]

pub async fn get_papers(
    query: String,
    category: String,
    date: String,
    end_date: String,
    page: usize,
    page_size: usize,
    negative_query: String,
    use_llm: bool,
) -> Result<Vec<Paper>, ServerFnError> {

    use chrono::DateTime;
    use chrono::Utc;
    use sqlx::sqlite::SqlitePool;

    let db_path = get_db_path().await?;

    let db_url = format!(
        "sqlite:{}?mode=rwc",
        db_path
    );

    let pool =
        SqlitePool::connect(&db_url)
            .await
            .map_err(|e| {

                ServerFnError::new(
                    format!(
            "Db connection error: {}",
            e
        ),
                )
            })?;

    ensure_schema(&pool).await?;

    let offset =
        (page.max(1) - 1) * page_size;

    log::info!(
        "Searching papers: \
         query='{}', cat='{}', \
         start='{}', end='{}', \
         page={}, size={}",
        query,
        category,
        date,
        end_date,
        page,
        page_size
    );

    let mut sql =
        "SELECT id, url, title, \
         updated, published, summary, \
         primary_category, \
         categories, authors, \
         pdf_link FROM papers"
            .to_string();

    let mut conditions = Vec::new();

    if !query.is_empty() {

        conditions.push(
            "id IN (SELECT id FROM \
             papers_fts WHERE \
             papers_fts MATCH ?)"
                .to_string(),
        );
    }

    if !category.is_empty()
        && category != "all"
    {

        conditions.push(
            "primary_category = ?"
                .to_string(),
        );
    }

    if !date.is_empty() {

        if !end_date.is_empty() {

            conditions.push(
                "date(published) \
                 BETWEEN ? AND ?"
                    .to_string(),
            );
        } else {

            conditions.push(
                "date(published) = ?"
                    .to_string(),
            );
        }
    }

    if !conditions.is_empty() {

        sql.push_str(" WHERE ");

        sql.push_str(
            &conditions.join(" AND "),
        );
    }

    sql.push_str(
        " ORDER BY published DESC \
         LIMIT ? OFFSET ?",
    );

    let mut q = sqlx::query(&sql);

    if !query.is_empty() {

        // Prepare FTS5 query: simple word match
        let fts_query = query
            .split_whitespace()
            .map(|w| format!("{}*", w))
            .collect::<Vec<_>>()
            .join(" ");

        q = q.bind(fts_query);
    }

    if !category.is_empty()
        && category != "all"
    {

        q = q.bind(category);
    }

    if !date.is_empty() {

        if !end_date.is_empty() {

            q = q
                .bind(date)
                .bind(end_date);
        } else {

            q = q.bind(date);
        }
    }

    q = q
        .bind(page_size as i64)
        .bind(offset as i64);

    let rows = q
        .fetch_all(&pool)
        .await
        .map_err(|e| {

            log::error!(
                "Database query \
                 error: {}",
                e
            );

            ServerFnError::new(format!(
                "Query error: {}",
                e
            ))
        })?;

    log::info!(
        "Found {} papers matching \
         criteria",
        rows.len()
    );

    use rayon::prelude::*;
    use sqlx::Row;

    let papers: Vec<Paper> = rows
        .into_par_iter()
        .map(|row| {
            let updated: DateTime<Utc> = row.try_get("updated").unwrap_or_else(|_| Utc::now());
            let published: DateTime<Utc> = row.try_get("published").unwrap_or_else(|_| Utc::now());

            Paper {
                id: row.get("id"),
                url: row.get::<Option<String>, _>("url").unwrap_or_default(),
                title: row.get::<Option<String>, _>("title").unwrap_or_default(),
                updated: updated.timestamp(),
                published: published.timestamp(),
                summary: row.get::<Option<String>, _>("summary").unwrap_or_default(),
                primary_category: row
                    .get::<Option<String>, _>("primary_category")
                    .unwrap_or_default(),
                categories: row
                    .get::<Option<String>, _>("categories")
                    .unwrap_or_default(),
                authors: row.get::<Option<String>, _>("authors").unwrap_or_default(),
                pdf_link: row.get("pdf_link"),
            }
        })
        .collect();

    #[cfg(feature = "ssr")]
    if !negative_query.is_empty() {
        // AI filtering is now handled progressively on the client
        // to prevent long blocking times.
        // The papers are returned "raw" and then filtered via stream.
    }

    Ok(papers)
}

#[server(GetPaperCount, "/api", input = Bitcode, output = Bitcode)]

pub async fn get_paper_count(
    query: String,
    category: String,
    date: String,
    end_date: String,
) -> Result<usize, ServerFnError> {

    use sqlx::Row;
    use sqlx::sqlite::SqlitePool;

    let db_path = get_db_path().await?;

    let db_url = format!(
        "sqlite:{}?mode=rwc",
        db_path
    );

    let pool =
        SqlitePool::connect(&db_url)
            .await
            .map_err(|e| {

                ServerFnError::new(
                    format!(
            "Db connection error: {}",
            e
        ),
                )
            })?;

    ensure_schema(&pool).await?;

    let mut sql = "SELECT COUNT(*) \
                   FROM papers"
        .to_string();

    let mut conditions = Vec::new();

    if !query.is_empty() {

        conditions.push(
            "id IN (SELECT id FROM \
             papers_fts WHERE \
             papers_fts MATCH ?)"
                .to_string(),
        );
    }

    if !category.is_empty()
        && category != "all"
    {

        conditions.push(
            "primary_category = ?"
                .to_string(),
        );
    }

    if !date.is_empty() {

        if !end_date.is_empty() {

            conditions.push(
                "date(published) \
                 BETWEEN ? AND ?"
                    .to_string(),
            );
        } else {

            conditions.push(
                "date(published) = ?"
                    .to_string(),
            );
        }
    }

    if !conditions.is_empty() {

        sql.push_str(" WHERE ");

        sql.push_str(
            &conditions.join(" AND "),
        );
    }

    let mut q = sqlx::query(&sql);

    if !query.is_empty() {

        let fts_query = query
            .split_whitespace()
            .map(|w| format!("{}*", w))
            .collect::<Vec<_>>()
            .join(" ");

        q = q.bind(fts_query);
    }

    if !category.is_empty()
        && category != "all"
    {

        q = q.bind(category);
    }

    if !date.is_empty() {

        if !end_date.is_empty() {

            q = q
                .bind(date)
                .bind(end_date);
        } else {

            q = q.bind(date);
        }
    }

    let row = q
        .fetch_one(&pool)
        .await
        .map_err(|e| {

            ServerFnError::new(format!(
                "Count error: {}",
                e
            ))
        })?;

    let count: i64 = row.get(0);

    Ok(count as usize)
}

#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    Default,
    PartialEq,
    bitcode::Encode,
    bitcode::Decode,
)]

pub struct ArxivConfig {
    pub category: String,
    pub start: i32,
    pub max_results: i32,
    pub lookback_days: i32,
}

#[allow(clippy::inline_always)]
#[inline(always)]
#[server(GetArxivConfig, "/api/get_arxiv_config", input = Bitcode, output = Bitcode)]

pub async fn get_arxiv_config()
-> Result<ArxivConfig, ServerFnError> {

    let content =
        std::fs::read_to_string(
            "config.toml",
        )
        .or_else(|_| {

            std::fs::read_to_string(
                "../config.toml",
            )
        })
        .map_err(|e| {

            ServerFnError::new(format!(
                "Failed to read \
                 config: {}",
                e
            ))
        })?;

    #[derive(Deserialize)]

    struct Config {
        arxiv: ArxivConfig,
    }

    let config: Config = toml::from_str(&content)
        .map_err(|e| ServerFnError::new(format!("Failed to parse config: {}", e)))?;

    Ok(config.arxiv)
}

#[allow(clippy::inline_always)]
#[inline(always)]
#[server(SaveArxivConfig, "/api/save_arxiv_config", input = Bitcode, output = Bitcode)]

pub async fn save_arxiv_config(
    password: String,
    arxiv: ArxivConfig,
) -> Result<(), ServerFnError> {

    if !verify_admin(&password).await? {

        return Err(
            ServerFnError::new(
                "Unauthorized: \
                 Invalid admin \
                 password",
            ),
        );
    }

    let path = if std::path::Path::new(
        "config.toml",
    )
    .exists()
    {

        "config.toml"
    } else {

        "../config.toml"
    };

    let content =
        std::fs::read_to_string(path)
            .map_err(|e| {

            ServerFnError::new(format!(
                "Failed to read \
                 config: {}",
                e
            ))
        })?;

    let mut config: toml::Value = content.parse().map_err(|e| {
        ServerFnError::new(format!("Failed to parse TOML: {}", e))
    })?;

    if let Some(arxiv_val) =
        config.get_mut("arxiv")
    {

        if let Some(table) =
            arxiv_val.as_table_mut()
        {

            table.insert(
                "category".to_string(),
                toml::Value::String(
                    arxiv.category,
                ),
            );

            table.insert(
                "start".to_string(),
                toml::Value::Integer(
                    arxiv.start as i64,
                ),
            );

            table.insert(
                "max_results"
                    .to_string(),
                toml::Value::Integer(
                    arxiv.max_results
                        as i64,
                ),
            );

            table.insert(
                "lookback_days"
                    .to_string(),
                toml::Value::Integer(
                    arxiv.lookback_days
                        as i64,
                ),
            );
        }
    }

    std::fs::write(
        path,
        config.to_string(),
    )
    .map_err(|e| {

        ServerFnError::new(format!(
            "Failed to write config: \
             {}",
            e
        ))
    })?;

    Ok(())
}

#[allow(clippy::inline_always)]
#[inline(always)]
#[server(GetConfig, "/api/get_config", input = Bitcode, output = Bitcode)]

pub async fn get_config()
-> Result<String, ServerFnError> {

    let content =
        std::fs::read_to_string(
            "config.toml",
        )
        .or_else(|_| {

            std::fs::read_to_string(
                "../config.toml",
            )
        })
        .map_err(|e| {

            ServerFnError::new(format!(
                "Failed to read \
                 config: {}",
                e
            ))
        })?;

    Ok(content)
}

#[cfg(feature = "ssr")]

async fn verify_admin(
    password: &str
) -> Result<bool, ServerFnError> {

    use argon2::Argon2;
    use argon2::password_hash::PasswordHash;
    use argon2::password_hash::PasswordVerifier;

    let assets_dir =
        if std::path::Path::new(
            "assets",
        )
        .exists()
        {

            "assets"
        } else {

            "../assets"
        };

    let hash_path =
        std::path::Path::new(
            assets_dir,
        )
        .join("admin.bin");

    let hash_str =
        std::fs::read_to_string(
            hash_path,
        )
        .map_err(|e| {

            ServerFnError::new(format!(
                "Security system \
                 error: {}",
                e
            ))
        })?;

    let parsed_hash =
        PasswordHash::new(&hash_str)
            .map_err(|e| {

                ServerFnError::new(
                    format!(
                        "Hash error: \
                         {}",
                        e
                    ),
                )
            })?;

    Ok(Argon2::default()
        .verify_password(
            password.as_bytes(),
            &parsed_hash,
        )
        .is_ok())
}

#[allow(clippy::inline_always)]
#[inline(always)]
#[server(VerifyAdmin, "/api/verify_admin", input = Bitcode, output = Bitcode)]

pub async fn verify_admin_password(
    password: String
) -> Result<bool, ServerFnError> {

    verify_admin(&password).await
}

#[server(UpdateAdminPassword, "/api/update_admin", input = Bitcode, output = Bitcode)]

pub async fn update_admin_password(
    old_password: String,
    new_password: String,
) -> Result<(), ServerFnError> {

    if !verify_admin(&old_password)
        .await?
    {

        return Err(
            ServerFnError::new(
                "Unauthorized: \
                 Invalid current \
                 password",
            ),
        );
    }

    use argon2::Argon2;
    use argon2::password_hash::PasswordHasher;
    use argon2::password_hash::SaltString;
    use argon2::password_hash::rand_core::OsRng;

    let salt = SaltString::generate(
        &mut OsRng,
    );

    let argon2 = Argon2::default();

    let password_hash = argon2
        .hash_password(
            new_password.as_bytes(),
            &salt,
        )
        .map_err(|e| {

            ServerFnError::new(format!(
                "Failed to hash \
                 password: {}",
                e
            ))
        })?
        .to_string();

    let assets_dir =
        if std::path::Path::new(
            "assets",
        )
        .exists()
        {

            "assets"
        } else {

            "../assets"
        };

    let admin_bin_path =
        std::path::Path::new(
            assets_dir,
        )
        .join("admin.bin");

    // Remove existing file first to
    // avoid appending to stale data
    if admin_bin_path.exists() {

        std::fs::remove_file(
            &admin_bin_path,
        )?;
    }

    std::fs::write(
        admin_bin_path,
        password_hash,
    )
    .map_err(|e| {

        ServerFnError::new(format!(
            "Failed to save password: \
             {}",
            e
        ))
    })?;

    Ok(())
}

#[server(SaveConfig, "/api/save_config", input = Bitcode, output = Bitcode)]

pub async fn save_config(
    password: String,
    content: String,
) -> Result<(), ServerFnError> {

    if !verify_admin(&password).await? {

        return Err(
            ServerFnError::new(
                "Unauthorized: \
                 Invalid admin \
                 password",
            ),
        );
    }

    let path = if std::path::Path::new(
        "config.toml",
    )
    .exists()
    {

        "config.toml"
    } else {

        "../config.toml"
    };

    std::fs::write(path, content)
        .map_err(|e| {

            ServerFnError::new(format!(
                "Failed to write \
                 config: {}",
                e
            ))
        })?;

    Ok(())
}

#[derive(
    Serialize,
    Deserialize,
    Encode,
    Decode,
    Clone,
    Debug,
    Default,
    PartialEq,
)]

pub struct VersionInfo {
    pub build_semver: String,
    pub authors: String,
    pub repository: String,
    pub build_timestamp: String,
    pub optimization: String,
    pub debug_symbols: String,
    pub target_triple: String,
    pub git_sha: String,
    pub git_branch: String,
    pub git_author: String,
    pub git_commit_message: String,
    pub rustc_semver: String,
    pub llvm_version: String,
    pub host_triple: String,
    pub os_name: String,
    pub os_version: String,
    pub kernel_version: String,
    pub cpu_brand: String,
    pub cpu_cores: String,
    pub total_memory: String,
}

pub static REPOSITORY: LazyLock<
    String,
> = LazyLock::new(|| {

    "https://github.com/panayang/arxiv-daily".to_string()
});

#[allow(clippy::inline_always)]
#[inline(always)]
#[server(GetVersionInfo, "/api", input = Bitcode, output = Bitcode)]

pub async fn get_version_info()
-> Result<VersionInfo, ServerFnError> {

    Ok(VersionInfo {
        build_semver: env!("CARGO_PKG_VERSION").to_string(),
        authors: env!("CARGO_PKG_AUTHORS").to_string(),
        repository: REPOSITORY.clone(),
        build_timestamp: env!("VERGEN_BUILD_TIMESTAMP").to_string(),
        optimization: env!("VERGEN_CARGO_OPT_LEVEL").to_string(),
        debug_symbols: env!("VERGEN_CARGO_DEBUG").to_string(),
        target_triple: env!("VERGEN_CARGO_TARGET_TRIPLE").to_string(),
        git_sha: env!("VERGEN_GIT_SHA").to_string(),
        git_branch: env!("VERGEN_GIT_BRANCH").to_string(),
        git_author: format!("{} <{}>", env!("VERGEN_GIT_COMMIT_AUTHOR_NAME"), env!("VERGEN_GIT_COMMIT_AUTHOR_EMAIL")),
        git_commit_message: env!("VERGEN_GIT_COMMIT_MESSAGE").to_string(),
        rustc_semver: env!("VERGEN_RUSTC_SEMVER").to_string(),
        llvm_version: env!("VERGEN_RUSTC_LLVM_VERSION").to_string(),
        host_triple: env!("VERGEN_RUSTC_HOST_TRIPLE").to_string(),
        os_name: env!("VERGEN_SYSINFO_NAME").to_string(),
        os_version: env!("VERGEN_SYSINFO_OS_VERSION").to_string(),
        kernel_version: env!("VERGEN_SYSINFO_KERNEL_VERSION").to_string(),
        cpu_brand: env!("VERGEN_SYSINFO_CPU_BRAND").to_string(),
        cpu_cores: env!("VERGEN_SYSINFO_CPU_CORE_COUNT").to_string(),
        total_memory: env!("VERGEN_SYSINFO_TOTAL_MEMORY").to_string(),
    })
}


#[server(FetchNewArticles, "/api", input = Bitcode, output = Bitcode)]

pub async fn fetch_new_articles(
    category: String,
    start_date: String,
    end_date: String,
) -> Result<usize, ServerFnError> {

    use chrono::Utc;
    use feed_rs::model::Entry;
    use sqlx::sqlite::SqlitePool;

    let db_path = get_db_path().await?;

    let db_url = format!(
        "sqlite:{}?mode=rwc",
        db_path
    );

    let pool =
        SqlitePool::connect(&db_url)
            .await
            .map_err(|e| {

                ServerFnError::new(
                    format!(
            "Db connection error: {}",
            e
        ),
                )
            })?;

    ensure_schema(&pool).await?;

    use serde::Deserialize;
    use toml;

    #[derive(Deserialize)]

    struct ArxivConfig {
        category: String,
        #[serde(default)]
        start: i32,
        #[serde(
            default = "default_max_results"
        )]
        max_results: i32,
        #[serde(
            default = "default_lookback"
        )]
        lookback_days: i32,
    }

    fn default_lookback() -> i32 {

        7
    }

    fn default_max_results() -> i32 {

        50
    }

    #[derive(Deserialize)]

    struct Config {
        arxiv: ArxivConfig,
    }

    let config_content =
        std::fs::read_to_string(
            "config.toml",
        )
        .or_else(|_| {

            std::fs::read_to_string(
                "../config.toml",
            )
        })
        .map_err(|e| {

            ServerFnError::new(format!(
                "Failed to read \
                 config: {}",
                e
            ))
        })?;

    let config: Config = toml::from_str(&config_content)
        .map_err(|e| ServerFnError::new(format!("Failed to parse config: {}", e)))?;

    let target_category =
        if category == "all" {

            config
                .arxiv
                .category
        } else {

            category
        };

    let url = if !end_date.is_empty() {

        let s = if start_date.is_empty()
        {

            &end_date
        } else {

            &start_date
        };

        let s_fmt =
            s.replace("-", "") + "0000";

        let e_fmt = end_date
            .replace("-", "")
            + "2359";

        format!(
            "https://export.arxiv.org/api/query?search_query=cat:{}+AND+submittedDate:[{}+TO+{}]&start={}&max_results={}&sortBy=submittedDate&sortOrder=descending",
            target_category, s_fmt, e_fmt, config.arxiv.start, config.arxiv.max_results
        )
    } else {

        let now = Utc::now();

        let start = now
            - chrono::Duration::days(
                config
                    .arxiv
                    .lookback_days
                    as i64,
            );

        let s_fmt = start
            .format("%Y%m%d%H%M")
            .to_string();

        let e_fmt = now
            .format("%Y%m%d%H%M")
            .to_string();

        format!(
            "https://export.arxiv.org/api/query?search_query=cat:{}+AND+submittedDate:[{}+TO+{}]&start={}&max_results={}&sortBy=submittedDate&sortOrder=descending",
            target_category, s_fmt, e_fmt, config.arxiv.start, config.arxiv.max_results
        )
    };

    let client = reqwest::Client::new();

    let retries = 5;

    let mut delay = tokio::time::Duration::from_secs(3);

    let mut response_bytes = None;

    for i in 0 .. retries {

        log::info!(
            "Fetching ArXiv articles \
             (attempt {}/{}): {}",
            i + 1,
            retries,
            url
        );

        let res = client
            .get(&url)
            .send()
            .await;

        match res {
            | Ok(resp) => {

                let status =
                    resp.status();

                if status.is_success() {

                    response_bytes =
                        Some(resp.bytes().await.map_err(|e| {
                            ServerFnError::new(format!("Failed to read body: {}", e))
                        })?);

                    break;
                } else if status
                    .as_u16()
                    == 429
                    || status.as_u16()
                        == 503
                {

                    if i == retries - 1
                    {

                        return Err(ServerFnError::new(format!(
                            "ArXiv API error ({}): Too many attempts",
                            status
                        )));
                    }

                    log::warn!(
                        "ArXiv API error ({}). Retrying in {}s...",
                        status,
                        delay.as_secs()
                    );

                    tokio::time::sleep(
                        delay,
                    )
                    .await;

                    delay *= 2;
                } else {

                    let err_msg = resp.text().await.unwrap_or_default();

                    return Err(ServerFnError::new(format!(
                        "ArXiv API error ({}): {}",
                        status, err_msg
                    )));
                }
            },
            | Err(e) => {

                if i == retries - 1 {

                    return Err(ServerFnError::new(format!(
                        "ArXiv request failed after {} attempts: {}",
                        retries, e
                    )));
                }

                log::warn!(
                    "ArXiv request \
                     error: {}. \
                     Retrying in \
                     {}s...",
                    e,
                    delay.as_secs()
                );

                tokio::time::sleep(
                    delay,
                )
                .await;

                delay *= 2;
            },
        }
    }

    let bytes = response_bytes
        .ok_or_else(|| {

            ServerFnError::new(
                "Failed to fetch \
                 articles after \
                 retries",
            )
        })?;

    let feed = feed_rs::parser::parse(
        &bytes[..],
    )
    .map_err(|e| {

        ServerFnError::new(format!(
            "Failed to parse feed: {}",
            e
        ))
    })?;

    let count = feed.entries.len();

    let papers: Vec<Paper> = feed
        .entries
        .into_iter()
        .map(|entry: Entry| {
            let authors: Vec<String> = entry.authors.iter().map(|a| a.name.clone()).collect();
            let pdf_link = entry
                .links
                .iter()
                .find(|l| l.media_type.as_deref() == Some("application/pdf"))
                .map(|l| l.href.clone());

            let raw_id = entry.id.clone();
            let parsed_id = if let Some(pos) = raw_id.find("/abs/") {
                let s = &raw_id[pos + 5..];
                if let Some(v_pos) = s.rfind('v') {
                    if s[v_pos + 1..].chars().all(|c| c.is_ascii_digit()) {
                        s[..v_pos].to_string()
                    } else {
                        s.to_string()
                    }
                } else {
                    s.to_string()
                }
            } else {
                raw_id.clone()
            };

            Paper {
                id: parsed_id,
                url: raw_id,
                title: entry.title.map(|t| t.content).unwrap_or_default(),
                updated: entry
                    .updated
                    .map(|d| d.timestamp())
                    .unwrap_or_else(|| Utc::now().timestamp()),
                published: entry
                    .published
                    .map(|d| d.timestamp())
                    .unwrap_or_else(|| Utc::now().timestamp()),
                summary: entry.summary.map(|s| s.content).unwrap_or_default(),
                primary_category: entry
                    .categories
                    .first()
                    .map(|c| c.term.clone())
                    .unwrap_or_default(),
                categories: entry
                    .categories
                    .iter()
                    .map(|c| c.term.clone())
                    .collect::<Vec<_>>()
                    .join(","),
                authors: serde_json::to_string(&authors).unwrap_or_default(),
                pdf_link,
            }
        })
        .collect();

    if !papers.is_empty() {

        // Bulk insert using a single query for better performance
        let mut query_builder: sqlx::QueryBuilder<'_, sqlx::Sqlite> = sqlx::QueryBuilder::new(
            "INSERT INTO papers (id, url, title, updated, published, summary, primary_category, categories, authors, pdf_link) ",
        );

        query_builder.push_values(papers, |mut b, paper| {
            b.push_bind(paper.id)
                .push_bind(paper.url)
                .push_bind(paper.title)
                .push_bind(
                    chrono::DateTime::from_timestamp(paper.updated, 0).unwrap_or_default(),
                )
                .push_bind(
                    chrono::DateTime::from_timestamp(paper.published, 0)
                        .unwrap_or_default(),
                )
                .push_bind(paper.summary)
                .push_bind(paper.primary_category)
                .push_bind(paper.categories)
                .push_bind(paper.authors)
                .push_bind(paper.pdf_link);
        });

        query_builder.push(
            " ON CONFLICT(id) DO \
             UPDATE SET
            updated = excluded.updated,
            title = excluded.title,
            summary = excluded.summary,
            url = excluded.url",
        );

        let query =
            query_builder.build();

        query
            .execute(&pool)
            .await
            .map_err(|e| {

                ServerFnError::new(
                    format!(
                        "Bulk insert \
                         error: {}",
                        e
                    ),
                )
            })?;
    }

    Ok(count)
}

#[cfg(feature = "ssr")]

async fn ensure_schema(
    pool: &sqlx::SqlitePool
) -> Result<(), ServerFnError> {

    // Ensure table exists
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS \
         papers (id TEXT PRIMARY KEY, \
         url TEXT, title TEXT, \
         updated DATETIME, published \
         DATETIME, summary TEXT, \
         primary_category TEXT, \
         categories TEXT, authors \
         TEXT, pdf_link TEXT)",
    )
    .execute(pool)
    .await
    .map_err(|e| {

        ServerFnError::new(format!(
            "Table creation error: {}",
            e
        ))
    })?;

    // FTS5 Virtual Table for searching
    sqlx::query(
        "CREATE VIRTUAL TABLE IF NOT \
         EXISTS papers_fts USING \
         fts5(id UNINDEXED, title, \
         summary, content='papers', \
         content_rowid='rowid')",
    )
    .execute(pool)
    .await
    .map_err(|e| {

        ServerFnError::new(format!(
            "FTS table creation \
             error: {}",
            e
        ))
    })?;

    // Triggers to keep FTS5 in sync
    sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS \
         papers_ai AFTER INSERT ON \
         papers BEGIN
                  INSERT INTO \
         papers_fts(rowid, id, title, \
         summary) VALUES (new.rowid, \
         new.id, new.title, \
         new.summary);
                END;",
    )
    .execute(pool)
    .await
    .ok();

    sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS \
         papers_ad AFTER DELETE ON \
         papers BEGIN
                  INSERT INTO \
         papers_fts(papers_fts, \
         rowid, id, title, summary) \
         VALUES('delete', old.rowid, \
         old.id, old.title, \
         old.summary);
                END;",
    )
    .execute(pool)
    .await
    .ok();

    sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS \
         papers_au AFTER UPDATE ON \
         papers BEGIN
                  INSERT INTO \
         papers_fts(papers_fts, \
         rowid, id, title, summary) \
         VALUES('delete', old.rowid, \
         old.id, old.title, \
         old.summary);
                  INSERT INTO \
         papers_fts(rowid, id, title, \
         summary) VALUES (new.rowid, \
         new.id, new.title, \
         new.summary);
                END;",
    )
    .execute(pool)
    .await
    .ok();

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS \
         idx_papers_published ON \
         papers (published DESC)",
    )
    .execute(pool)
    .await
    .ok();

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS \
         idx_papers_primary_category \
         ON papers (primary_category)",
    )
    .execute(pool)
    .await
    .ok();

    // Initial population of FTS if empty
    let count: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM \
         papers_fts",
    )
    .fetch_one(pool)
    .await
    .unwrap_or((0,));

    if count.0 == 0 {

        sqlx::query(
            "INSERT INTO \
             papers_fts(rowid, id, \
             title, summary) SELECT \
             rowid, id, title, \
             summary FROM papers",
        )
        .execute(pool)
        .await
        .ok();
    }

    Ok(())
}

#[cfg(feature = "ssr")]

async fn get_db_path()
-> Result<String, ServerFnError> {

    use serde::Deserialize;
    use toml;

    #[derive(Deserialize)]

    struct Config {
        database: DatabaseConfig,
    }

    #[derive(Deserialize)]

    struct DatabaseConfig {
        path: String,
    }

    let config_content =
        std::fs::read_to_string(
            "config.toml",
        )
        .or_else(|_| {

            std::fs::read_to_string(
                "../config.toml",
            )
        })
        .map_err(|e| {

            ServerFnError::new(format!(
                "Failed to read \
                 config: {}",
                e
            ))
        })?;

    let config: Config = toml::from_str(&config_content)
        .map_err(|e| ServerFnError::new(format!("Failed to parse config: {}", e)))?;

    Ok(config.database.path)
}

#[cfg(feature = "ssr")]
mod ai;

mod surprise;

// Restore generation tracking for explicit invalidation
#[cfg(feature = "ssr")]

static MODEL_GENERATION:
    std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(
        0,
    );

#[cfg(feature = "ssr")]

static MODEL: Mutex<
    Option<
        Arc<
            std::sync::Mutex<
                ai::Gemma3,
            >,
        >,
    >,
> = Mutex::const_new(None);

#[cfg(feature = "ssr")]

static ACTIVE_MODEL_WEAK: Mutex<
    Option<
        std::sync::Weak<
            std::sync::Mutex<
                ai::Gemma3,
            >,
        >,
    >,
> = Mutex::const_new(None);


#[server(UnloadModel, "/api/unload_model", input = Bitcode, output = Bitcode)]

pub async fn unload_model()
-> Result<(), ServerFnError> {

    log::info!(
        "Request to unload AI model..."
    );

    let mut model = MODEL.lock().await;

    // Always increment generation to invalidate streams immediately
    let old_gen = MODEL_GENERATION.load(std::sync::atomic::Ordering::Relaxed);

    let new_gen = MODEL_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

    log::info!(
        "🔄 Generation incremented: \
         {} -> {}",
        old_gen,
        new_gen
    );

    if model.is_some() {

        // Take ownership to get strong count before dropping
        let model_arc = model.take();

        if let Some(arc) = model_arc {

            let strong_count =
                Arc::strong_count(&arc);

            log::info!(
                "📊 Model Arc strong \
                 count before drop: {}",
                strong_count
            );

            // Explicitly drop the Arc
            drop(arc);
        }

        // Also clear the weak reference tracker
        let mut weak_lock =
            ACTIVE_MODEL_WEAK
                .lock()
                .await;

        *weak_lock = None;

        drop(weak_lock);

        // Release the MODEL lock to allow other tasks to see None
        drop(model);

        // Give a moment for in-flight spawn_blocking tasks to finish and drop their references
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        log::info!(
            "✅ AI Model dropped from \
             global storage. Memory \
             should be freed shortly."
        );
    } else {

        log::info!(
            "AI Model was already not \
             loaded."
        );
    }

    Ok(())
}

use leptos::server_fn::codec::ByteStream;

#[server(StreamAiFilter, "/api/stream_filter", input=Bitcode, output=Streaming)]

pub async fn stream_ai_filter(
    papers: Vec<Paper>,
    negative_query: String,
) -> Result<ByteStream, ServerFnError> {

    #[cfg(feature = "ssr")]
    {

        use std::sync::Arc;
        use std::sync::atomic::Ordering;

        use bytes::Bytes;
        use futures::StreamExt;

        // Fast path check
        if negative_query.is_empty() {

            return Ok(ByteStream::new(futures::stream::iter::<Vec<Result<Bytes, ServerFnError>>>(vec![])));
        }

        log::info!(
            "Streaming AI filter for \
             {} papers with query: \
             '{}'",
            papers.len(),
            negative_query
        );

        let (model_weak, current_gen) = {

            let mut model_lock =
                MODEL.lock().await;

            let r#gen =
                MODEL_GENERATION.load(
                    Ordering::Relaxed,
                );

            log::info!(
                "📡 Stream starting \
                 with generation: {}",
                r#gen
            );

            if let Some(m) =
                &*model_lock
            {

                (
                    Arc::downgrade(m),
                    r#gen,
                )
            } else {

                // Check if an old model is still lingering in memory (phantom model)
                let mut cleanup_retries =
                    0;

                loop {

                    let weak_lock = ACTIVE_MODEL_WEAK.lock().await;

                    if let Some(weak) =
                        &*weak_lock
                    {

                        if weak
                            .upgrade()
                            .is_some()
                        {

                            if cleanup_retries > 10 {
                                 // After ~2 seconds, give up
                                 return Err(ServerFnError::new("System busy: Previous model is still unloading. Please retry in a few seconds."));
                             }

                            log::warn!("Old model still active (memory cleanup pending). Waiting...");

                            drop(weak_lock); // Release lock
                            drop(model_lock); // Release main lock to allow other threads to progress
                            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

                            // Re-acquire main lock
                            model_lock = MODEL.lock().await;

                            // Check if someone else loaded it while we slept
                            #[allow(unused_variables)]
                            if let Some(m) = &*model_lock {
                                 // Model was loaded by another request while we waited. Use it.
                                 // Break out of retry loop - we'll use this model
                                 break;
                             }

                            cleanup_retries += 1;

                            continue; // Continue the loop to re-check
                        }
                    }

                    break; // No active weak ref or weak ref is dead, proceed to load
                }

                // After potential waiting, re-check if model was loaded by another request
                if let Some(m) =
                    &*model_lock
                {

                    (Arc::downgrade(m), MODEL_GENERATION.load(Ordering::Relaxed))
                } else {

                    let (model_path, tokenizer_path) = if std::path::Path::new("assets/llm.bin").exists() {
                        let tok_path = if std::path::Path::new("assets/tokenizer.json").exists() {
                            "assets/tokenizer.json".to_string()
                        } else {
                             "assets/tokenizer.bin".to_string()
                        };
                        ("assets/llm.bin".to_string(), tok_path)
                    } else if std::path::Path::new("../assets/llm.bin").exists() {
                        let tok_path = if std::path::Path::new("../assets/tokenizer.json").exists() {
                            "../assets/tokenizer.json".to_string()
                        } else {
                             "../assets/tokenizer.bin".to_string()
                        };
                        ("../assets/llm.bin".to_string(), tok_path)
                    } else {
                        ("assets/llm.bin".to_string(), "assets/tokenizer.json".to_string())
                    };

                    log::info!(
                        "Loading custom \
                         Gemma 3 model \
                         from {}... \
                         (spawn_blocking)",
                        model_path
                    );

                    // Offload loading to blocking thread
                    let gemma_res = tokio::task::spawn_blocking(move || {
                        let mut gemma = ai::Gemma3::new(model_path, tokenizer_path)
                            .map_err(|e| format!("Failed to load Gemma 3 model: {}", e))?;
                        gemma.set_initialized(true);
                        // Do a quick self-test (Only when debugging)
                        if cfg!(debug_assertions) {
                            gemma.self_test().map_err(|e| format!("{}", e))?;
                        }
                        Ok::<ai::Gemma3, String>(gemma)
                    }).await.map_err(|e| ServerFnError::new(format!("Join error: {}", e)))?;

                    let gemma =
                        match gemma_res
                        {
                            | Ok(g) => {
                                g
                            },
                            | Err(
                                e,
                            ) => {

                                log::error!("❌ Model loading or self-test failed: {}", e);

                                return Err(ServerFnError::new(e));
                            },
                        };

                    let m = Arc::new(std::sync::Mutex::new(gemma));

                    *model_lock =
                        Some(m.clone());

                    // Update Active Weak Ref
                    let mut weak_lock = ACTIVE_MODEL_WEAK.lock().await;

                    *weak_lock = Some(
                        Arc::downgrade(
                            &m,
                        ),
                    );

                    (
                        Arc::downgrade(
                            &m,
                        ),
                        r#gen,
                    )
                }
            }
        };

        let stream = futures::stream::iter(papers)
             .then(move |paper| {
                let model_weak = model_weak.clone();
                let negative_query = negative_query.clone();
                let params_gen = current_gen;

                async move {
                    // 1. Check generation
                    let current = MODEL_GENERATION.load(Ordering::Relaxed);
                    if current != params_gen {
                         log::info!("⏹️ Stream stopping: gen {} != {}", current, params_gen);
                         return Ok(FilterStreamMessage::Stopped);
                    }

                    // 2. Try to upgrade Weak
                    let model_arc = match model_weak.upgrade() {
                        Some(arc) => arc,
                        None => return Ok(FilterStreamMessage::Stopped),
                    };

                    // 3. Yield to prevent CPU starvation on single-threaded runtimes or heavy load
                    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;

                    let summary_snippet = if paper.summary.len() > 300 {
                        format!("{}...", &paper.summary[0..300])
                    } else {
                        paper.summary.clone()
                    };

                    let prompt = format!(
                        "<start_of_turn>user\nInstructions: Respond ONLY with YES or NO.\nNegative Filter: Skip papers related to \"{}\".\nPaper Title: {}\nSummary Snippet: {}\nQuestion: Should I skip this paper?\nAnswer: <end_of_turn>\n<start_of_turn>model\n",
                        negative_query, paper.title, summary_snippet
                    );

                    // Offload blocking inference
                    let result = tokio::task::spawn_blocking(move || {
                        let mut model = model_arc.lock().map_err(|_| "Poisoned lock")?;

                        // Check generation inside inference loop
                        // Use a non-move closure to ensure we read MODEL_GENERATION fresh each time
                        let check_cancel = || {
                            let current = MODEL_GENERATION.load(Ordering::Relaxed);
                            let cancelled = current != params_gen;
                            if cancelled {
                                log::info!("🛑 Cancellation detected: gen {} != {}", current, params_gen);
                            }
                            cancelled
                        };

                        model.complete(&prompt, 3, check_cancel).map_err(|e| e.to_string())
                    }).await;

                    let response = match result {
                        Ok(Ok(r)) => r,
                        Ok(Err(e)) => {
                             log::error!("AI completion internal error: {}", e);
                             return Ok(FilterStreamMessage::Result(paper.id, true));
                        }
                        Err(e) => {
                            log::error!("AI task join error: {}", e);
                             return Ok(FilterStreamMessage::Result(paper.id, true));
                        }
                    };

                    let response = response.trim().to_uppercase();
                    let is_excluded = response.contains("YES");

                    Ok(FilterStreamMessage::Result(paper.id, !is_excluded))
                }
            })
            // Remove take_while so we can send Stopped message
            .map(|result| {
                 match result {
                     Ok(msg) => {
                         let config = config::standard();
                         let bytes: Vec<u8> = bincode_next::encode_to_vec(&msg, config)
                            .map_err(|e| ServerFnError::new(e.to_string()))?;
                         Ok(Bytes::from(bytes))
                     }
                     Err(e) => Err(e),
                 }
            });

        return Ok(ByteStream::new(
            stream,
        ));
    }

    #[cfg(not(feature = "ssr"))]
    unreachable!(
        "stream_ai_filter should only \
         be called on server via \
         server function"
    )
}


fn html_escape(text: &str) -> String {

    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn fix_latex_symbols(
    latex: &str
) -> String {

    latex
        .replace(r"\,", " ") // Handle LaTeX small space
        .replace(r"\le", r"\leq") // Normalize inequalities
        .replace(r"\ge", r"\geq")
        .replace(r"\ast", "*") // Handle asterisk
        .replace(r"\star", "*")
        .replace(r"\mathcal", "") // Strip unsupported font commands
        .replace(r"\cal", "")
        .replace(r"\mathrm", "")
        .replace(r"\mathfrak", "")
        .replace(r"\mathbb", "")
        .replace(r"\mathsf", "")
        .replace(r"\mathtt", "")
        .replace(r"\rm", "")
        .replace(r"\it", "")
        .replace(r"\bf", "")
        .replace(r"\sf", "")
        .replace(r"\tt", "")
        .replace(r"\AA", "A")
        .replace(r"\BB", "B")
        .replace(r"\CC", "C")
        .replace(r"\DD", "D")
        .replace(r"\EE", "E")
        .replace(r"\FF", "F")
        .replace(r"\GG", "G")
        .replace(r"\HH", "H")
        .replace(r"\II", "I")
        .replace(r"\JJ", "J")
        .replace(r"\KK", "K")
        .replace(r"\LL", "L")
        .replace(r"\MM", "M")
        .replace(r"\NN", "N")
        .replace(r"\OO", "O")
        .replace(r"\PP", "P")
        .replace(r"\QQ", "Q")
        .replace(r"\RR", "R")
        .replace(r"\SS", "S")
        .replace(r"\TT", "T")
        .replace(r"\UU", "U")
        .replace(r"\VV", "V")
        .replace(r"\WW", "W")
        .replace(r"\XX", "X")
        .replace(r"\YY", "Y")
        .replace(r"\ZZ", "Z")
        .replace(r"\mathbf", r"\vec") // Fallback for bold vector if needed, or keeping it
        .replace('Γ', r"\Gamma")
        .replace('Δ', r"\Delta")
        .replace('Θ', r"\Theta")
        .replace('Λ', r"\Lambda")
        .replace('Ξ', r"\Xi")
        .replace('Π', r"\Pi")
        .replace('Σ', r"\Sigma")
        .replace('Φ', r"\Phi")
        .replace('Ψ', r"\Psi")
        .replace('Ω', r"\Omega")
        .replace('α', r"\alpha")
        .replace('β', r"\beta")
        .replace('γ', r"\gamma")
        .replace('δ', r"\delta")
        .replace('ε', r"\epsilon")
        .replace('ζ', r"\zeta")
        .replace('η', r"\eta")
        .replace('θ', r"\theta")
        .replace('ι', r"\iota")
        .replace('κ', r"\kappa")
        .replace('λ', r"\lambda")
        .replace('μ', r"\mu")
        .replace('ν', r"\nu")
        .replace('ξ', r"\xi")
        .replace('π', r"\pi")
        .replace('ρ', r"\rho")
        .replace('σ', r"\sigma")
        .replace('τ', r"\tau")
        .replace('υ', r"\upsilon")
        .replace('φ', r"\phi")
        .replace('χ', r"\chi")
        .replace('ψ', r"\psi")
        .replace('ω', r"\omega")
        .replace('≡', r"\equiv")
        .replace('±', r"\pm")
        .replace('×', r"\times")
        .replace('÷', r"\div")
        .replace('≈', r"\approx")
        .replace('≠', r"\neq")
        .replace('≤', r"\leq")
        .replace('≥', r"\geq")
        .replace('≦', r"\leq")
        .replace('≧', r"\geq")
        .replace('≃', r"\simeq")
        .replace('⋯', r"\cdots")
        .replace('…', r"\dots")
        .replace('∞', r"\infty")
        .replace('→', r"\to")
        .replace('∂', r"\partial")
        .replace('∇', r"\nabla")
}

fn try_latex_to_mathml(
    latex: &str,
    style: DisplayStyle,
) -> String {

    let mut current_latex =
        latex.to_string();

    for _ in 0 .. 15 {

        let result = latex_to_mathml(
            &current_latex,
            style,
        );

        // Handle successful result but check if it contains embedded error messages
        if let Ok(ref mathml) = result {

            if !mathml
                .contains("PARSE ERROR")
                && !mathml.contains(
                    "Undefined",
                )
            {

                return mathml
                    .to_string();
            }
            // If it contains an error message, treat it as an error and try to fix it
        }

        let err_str = match result {
            | Ok(ref s) => s.clone(),
            | Err(ref e) => {

                format!("{:?}", e)
            },
        };

        // 1. Extract command name from Undefined("Command(\"name\")") or PARSE ERROR blocks
        if let Some(pos) =
            err_str.find("Command(\"")
        {

            let cmd_part =
                &err_str[pos + 9 ..];

            let end = cmd_part
                .find("\\\"")
                .or_else(|| {

                    cmd_part.find("\")")
                });

            if let Some(end_idx) = end {

                let cmd = &cmd_part
                    [.. end_idx];

                if !cmd.is_empty() {

                    let to_replace = format!(
                        "\\{}",
                        cmd
                    );

                    let next =
                        current_latex
                            .replace(
                            &to_replace,
                            "",
                        );

                    if next
                        != current_latex
                    {

                        current_latex =
                            next;

                        continue;
                    }
                }
            }
        }

        // 2. Specialized symbol/structural errors
        if err_str
            .contains("Circumflex")
            || err_str
                .contains("Circumflex")
        {

            let next = current_latex
                .replace('^', "");

            if next != current_latex {

                current_latex = next;

                continue;
            }
        }

        if err_str
            .contains("Underscore")
        {

            let next = current_latex
                .replace('_', "");

            if next != current_latex {

                current_latex = next;

                continue;
            }
        }

        // 3. General Undefined fallback
        if let Some(pos) =
            err_str.find("Undefined(\"")
        {

            let sym_part =
                &err_str[pos + 11 ..];

            if let Some(end) =
                sym_part.find("\")")
            {

                let sym =
                    &sym_part[.. end];

                if !sym.is_empty() {

                    let to_replace = format!(
                        "\\{}",
                        sym
                    );

                    let next =
                        current_latex
                            .replace(
                            &to_replace,
                            "",
                        );

                    if next
                        != current_latex
                    {

                        current_latex =
                            next;

                        continue;
                    }
                }
            }
        }

        // 4. Force strip any remaining backslashed words if we're stuck
        // This is a "nuclear option" for unknown commands
        if let Some(pos) =
            current_latex.find('\\')
        {

            let after_slash =
                &current_latex
                    [pos + 1 ..];

            let end = after_slash
                .find(|c: char| {

                    !c.is_alphabetic()
                })
                .unwrap_or(
                    after_slash.len(),
                );

            if end > 0 {

                let to_remove = format!(
                    "\\{}",
                    &after_slash
                        [.. end]
                );

                current_latex =
                    current_latex
                        .replace(
                            &to_remove,
                            "",
                        );

                continue;
            }
        }

        break;
    }

    // Final Fallback: Display the original source
    let delimiter = match style {
        | DisplayStyle::Block => "$$",
        | DisplayStyle::Inline => "$",
    };

    format!(
        "{}{}{}",
        delimiter,
        html_escape(latex),
        delimiter
    )
}

pub fn render_math(
    text: &str
) -> String {

    static MATH_DISPLAY: LazyLock<
        Regex,
    > = LazyLock::new(|| {

        Regex::new(r"(?s)\$\$(.*?)\$\$")
            .unwrap()
    });

    static MATH_INLINE: LazyLock<
        Regex,
    > = LazyLock::new(|| {

        Regex::new(r"(?s)\$(.*?)\$")
            .unwrap()
    });

    // Placeholder system for hybrid text/MathML
    let mut placeholders = Vec::new();

    let mut result = text.to_string();

    // 1. Display math
    result = MATH_DISPLAY
        .replace_all(&result, |caps: &regex::Captures| {
            let latex = fix_latex_symbols(caps[1].trim());
            let rendered = try_latex_to_mathml(&latex, DisplayStyle::Block);
            let idx = placeholders.len();
            placeholders.push(rendered);
            format!("__MATH_P_{}__", idx)
        })
        .to_string();

    // 2. Inline math
    result = MATH_INLINE
        .replace_all(&result, |caps: &regex::Captures| {
            let latex = fix_latex_symbols(caps[1].trim());
            let rendered = try_latex_to_mathml(&latex, DisplayStyle::Inline);
            let idx = placeholders.len();
            placeholders.push(rendered);
            format!("__MATH_P_{}__", idx)
        })
        .to_string();

    // 3. Escape skeleton
    let mut final_result =
        html_escape(&result);

    // 4. Re-inject
    for (idx, rendered) in placeholders
        .into_iter()
        .enumerate()
    {

        let placeholder = format!(
            "__MATH_P_{}__",
            idx
        );

        final_result = final_result
            .replace(
                &placeholder,
                &rendered,
            );
    }

    final_result
}

#[allow(clippy::inline_always)]
#[inline(always)]
#[cfg(feature = "ssr")]
#[allow(dead_code)]

fn filter_with_keywords(
    papers: Vec<Paper>,
    negative_query: String,
) -> Vec<Paper> {

    if negative_query.is_empty()
        || papers.is_empty()
    {

        return papers;
    }

    let keywords: Vec<String> =
        negative_query
            .split(',')
            .map(|s| {

                s.trim()
                    .to_lowercase()
            })
            .filter(|s| !s.is_empty())
            .collect();

    if keywords.is_empty() {

        return papers;
    }

    log::info!(
        "Keyword filtering {} papers \
         with keywords: {:?}",
        papers.len(),
        keywords
    );

    papers
        .into_iter()
        .filter(|paper| {

            let title = paper
                .title
                .to_lowercase();

            let summary = paper
                .summary
                .to_lowercase();

            !keywords
                .iter()
                .any(|kw| {

                    title.contains(kw)
                        || summary
                            .contains(
                                kw,
                            )
                })
        })
        .collect()
}

pub fn shell(
    options: LeptosOptions
) -> impl IntoView {

    view! {
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width, initial-scale=1"/>
                <AutoReload options=options.clone()/>
                <HydrationScripts options/>
                <MetaTags/>
            </head>
            <body>
                <App/>
            </body>
        </html>
    }
}

/// Health check request for the web
/// worker
#[derive(
    Clone,
    serde::Serialize,
    serde::Deserialize,
)]

pub struct HealthCheckRequest {
    /// The origin URL to check
    pub origin: String,
}

/// Health check response from the web
/// worker
#[derive(
    Clone,
    serde::Serialize,
    serde::Deserialize,
)]

pub struct HealthCheckResponse {
    /// Whether the backend is reachable
    pub is_healthy: bool,
}

/// Runs in a dedicated Web Worker thread
/// via leptos_workers, keeping the main
/// UI thread completely unblocked.
#[cfg(feature = "hydrate")]
#[leptos_workers::worker(
    HealthCheckWorker
)]

pub async fn health_check_worker(
    req: HealthCheckRequest
) -> HealthCheckResponse {

    let url = format!(
        "{}/api/health",
        req.origin
    );

    let healthy =
        match gloo_net::http::Request::get(
            &url,
        )
        .send()
        .await
        {

            Ok(resp) => resp.ok(),
            Err(_) => false,
        };

    HealthCheckResponse {
        is_healthy: healthy,
    }
}

#[component]

fn HealthBanner() -> impl IntoView {

    let (
        backend_down,
        set_backend_down,
    ) = signal(false);

    #[cfg(feature = "hydrate")]
    {

        use leptos::task::spawn_local;

        spawn_local(async move {

            loop {

                let origin =
                    web_sys::window()
                        .expect("window")
                        .location()
                        .origin()
                        .unwrap_or_default();

                match health_check_worker(
                    HealthCheckRequest {
                        origin,
                    },
                )
                .await
                {

                    Ok(resp) => {

                        set_backend_down.set(
                            !resp.is_healthy,
                        );
                    },
                    Err(_) => {

                        set_backend_down
                            .set(true);
                    },
                }

                gloo_timers::future::TimeoutFuture::new(
                    5_000,
                )
                .await;
            }
        });
    }

    view! {
        <Show when=move || backend_down.get()>
            <div
                id="health-banner"
                class="fixed top-0 left-0 right-0 z-[9999] \
                       bg-obsidian-sidebar/90 backdrop-blur-xl \
                       border-b border-red-500/20 \
                       text-obsidian-heading text-center \
                       py-3 px-4 font-medium \
                       shadow-[0_4px_24px_rgba(239,68,68,0.1)] \
                       animate-fade-in"
            >
                <div class="flex items-center justify-center gap-2">
                    <span class="inline-block w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                    <span class="text-sm tracking-wide text-obsidian-text/70">
                        "Backend is unreachable — please check if the server is running"
                    </span>
                </div>
            </div>
        </Show>
    }
}

#[component]

pub fn App() -> impl IntoView {

    provide_meta_context();

    view! {
        <Stylesheet id="leptos" href="/pkg/web_app.css"/>
        <Title text="arXiv Dashboard"/>

        <HealthBanner/>
        <Router>
            <main class="min-h-screen bg-obsidian-bg text-obsidian-text font-sans selection:bg-obsidian-accent/30">
                <Routes fallback=|| "Not Found".into_any()>
                    <Route path=path!("") view=Dashboard/>
                </Routes>
            </main>
        </Router>
    }
}

#[component]

fn Dashboard() -> impl IntoView {

    let (input_val, set_input_val) =
        signal("".to_string());

    let (
        selected_category,
        set_selected_category,
    ) = signal("all".to_string());

    let (date_filter, set_date_filter) =
        signal("".to_string());

    let (
        end_date_filter,
        set_end_date_filter,
    ) = signal("".to_string());

    let (
        negative_query,
        set_negative_query,
    ) = signal("".to_string());

    let (
        show_fetch_status,
        set_show_fetch_status,
    ) = signal(false);

    let use_llm = RwSignal::new(true);

    let (page, set_page) =
        signal(1usize);

    let (show_config, set_show_config) =
        signal(false);

    let (show_about, set_show_about) =
        signal(false);

    let (
        show_surprise,
        set_show_surprise,
    ) = signal(false);

    let (
        hidden_paper_ids,
        set_hidden_paper_ids,
    ) = signal(
        std::collections::HashSet::new(
        ),
    );

    let (view_mode, set_view_mode) =
        signal("card".to_string());

    #[derive(
        Clone, Default, PartialEq,
    )]

    struct SearchParams {
        query: String,
        category: String,
        date: String,
        end_date: String,
        page: usize,
        page_size: usize,
        negative_query: String,
        use_llm: bool,
    }

    let page_size =
        RwSignal::new(51usize);

    let (
        trigger_search,
        set_trigger_search,
    ) = signal(SearchParams {
        query: "".to_string(),
        category: "all".to_string(),
        date: "".to_string(),
        end_date: "".to_string(),
        page: 1,
        page_size: 51,
        negative_query: "".to_string(),
        use_llm: use_llm
            .get_untracked(),
    });

    let papers = Resource::new(
        move || trigger_search.get(),
        |params| {

            async move {

                get_papers(
                    params.query,
                    params.category,
                    params.date,
                    params.end_date,
                    params.page,
                    params.page_size,
                    params
                        .negative_query,
                    params.use_llm,
                )
                .await
            }
        },
    );

    let total_count = Resource::new(
        move || {

            let p =
                trigger_search.get();

            (
                p.query,
                p.category,
                p.date,
                p.end_date,
            )
        },
        |(
            query,
            category,
            date,
            end_date,
        )| {

            async move {

                get_paper_count(
                    query,
                    category,
                    date,
                    end_date,
                )
                .await
            }
        },
    );

    // Live search debounce effect
    Effect::new(move |_| {

        let query = input_val.get();

        let timeout = set_timeout_with_handle(
            move || {
                set_trigger_search.update(|p| {
                    if p.query != query {
                        p.query = query;
                        p.page = 1;
                    }
                });
            },
            std::time::Duration::from_millis(300),
        );

        move || {
            if let Ok(timeout) = timeout
            {

                timeout.clear();
            }
        }
    });

    // Negative query debounce effect
    Effect::new(move |_| {

        let neg_query =
            negative_query.get();

        let timeout = set_timeout_with_handle(
            move || {
                set_trigger_search.update(|p| {
                    if p.negative_query != neg_query {
                        p.negative_query = neg_query;
                        p.page = 1;
                    }
                });
            },
            std::time::Duration::from_millis(1000),
        );

        move || {
            if let Ok(timeout) = timeout
            {

                timeout.clear();
            }
        }
    });

    // Auto-search on category/date/page_size change
    Effect::new(move |_| {

        let category =
            selected_category.get();

        let date = date_filter.get();

        let end_date =
            end_date_filter.get();

        let size = page_size.get();

        set_trigger_search.update(
            |p| {

                p.category = category;

                p.date = date;

                p.end_date = end_date;

                p.page_size = size;

                p.page = 1; // Reset page on filter change
            },
        );

        set_page.set(1);
    });

    // Client-side AI streaming effect
    Effect::new(move |_| {

        let papers_result =
            papers.get();

        let neg_query =
            negative_query.get();

        let use_llm_flag =
            use_llm.get();

        if let Some(Ok(
            current_papers,
        )) = papers_result
        {

            // Only stream if we have papers, a negative query, and LLM is enabled
            if !neg_query.is_empty()
                && use_llm_flag
            {

                spawn_local(
                    async move {

                        // Start fresh
                        set_hidden_paper_ids.set(std::collections::HashSet::new());

                        match stream_ai_filter(current_papers, neg_query).await {
                        Ok(byte_stream) => {
                            use futures::StreamExt;
                            // Unwrap the ByteStream newtype to get the actual stream
                            let mut stream = byte_stream.into_inner();

                            while let Some(chunk_res) = stream.next().await {
                                match chunk_res {
                                    Ok(chunk) => {
                                        // chunk should be Bytes
                                        let config = config::standard();
                                        if let Ok((msg, _size)) = bincode_next::decode_from_slice::<FilterStreamMessage, _>(&chunk, config) {
                                            match msg {
                                                FilterStreamMessage::Result(id, keep) => {
                                                    if !keep {
                                                        set_hidden_paper_ids.update(|set| {
                                                            set.insert(id);
                                                        });
                                                    }
                                                },
                                                FilterStreamMessage::Stopped => {
                                                    log::info!("AI Stream stopped explicitly.");
                                                    // Optional: Toast or UI indicator
                                                },
                                                FilterStreamMessage::Error(e) => {
                                                    log::error!("AI Stream error: {}", e);
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => log::error!("Stream error: {}", e),
                                }
                            }
                        }
                        Err(e) => log::error!("Streaming filter failed: {}", e),
                    }
                    },
                );
            } else {

                set_hidden_paper_ids.set(std::collections::HashSet::new());
            }
        }
    });

    let unload_action =
        Action::new(|_| {

            async move {

                unload_model().await
            }
        });

    let fetch_action = Action::new(
        move |(cat, start, end): &(
            String,
            String,
            String,
        )| {

            let cat = cat.clone();

            let start = start.clone();

            let end = end.clone();

            async move {

                let res =
                    fetch_new_articles(
                        cat.clone(),
                        start.clone(),
                        end.clone(),
                    )
                    .await;

                if res.is_ok() {

                    // After fetching, ensure we trigger a refetch of the papers resource
                    // even if the parameters (cat/start/end) are identical to current view
                    papers.refetch();

                    set_trigger_search.set(SearchParams {
                    query: "".to_string(),
                    category: cat,
                    date: start,
                    end_date: end,
                    page: 1,
                    page_size: page_size.get(),
                    negative_query: "".to_string(),
                    use_llm: true,
                });

                    set_page.set(1);
                }

                res
            }
        },
    );

    Effect::new(move |_| {

        let pending = fetch_action
            .pending()
            .get();

        let value = fetch_action
            .value()
            .get();

        if pending {

            set_show_fetch_status
                .set(true);
        } else if value.is_some() {

            set_timeout(
                move || {
                    set_show_fetch_status.set(false);
                },
                std::time::Duration::from_secs(4),
            );
        } else {

            set_show_fetch_status
                .set(false);
        }
    });

    let on_search = move |_| {

        set_trigger_search.set(
            SearchParams {
                query: input_val.get(),
                category:
                    selected_category
                        .get(),
                date: date_filter.get(),
                end_date:
                    end_date_filter
                        .get(),
                page: 1,
                page_size: page_size.get(),
                negative_query: negative_query.get(),
                use_llm: use_llm.get(),
            },
        );

        set_page.set(1);
    };

    let on_reset = move |_| {

        set_input_val
            .set("".to_string());

        set_selected_category
            .set("all".to_string());

        set_date_filter
            .set("".to_string());

        set_end_date_filter
            .set("".to_string());

        set_negative_query
            .set("".to_string());

        set_page.set(1);

        set_trigger_search.set(
            SearchParams {
                query: "".to_string(),
                category: "all"
                    .to_string(),
                date: "".to_string(),
                end_date: ""
                    .to_string(),
                page: 1,
                page_size: 51,
                negative_query: ""
                    .to_string(),
                use_llm: true,
            },
        );
    };

    // Global Hotkeys
    #[cfg(feature = "hydrate")]
    {

        use leptos::ev;
        use leptos::prelude::window_event_listener;

        let total_count_resource =
            total_count;

        window_event_listener(
            ev::keydown,
            move |ev| {

                let key = ev.key();

                // Modal closing with Escape
                if key == "Escape" {

                    if show_config.get()
                    {

                        set_show_config
                            .set(false);

                        return;
                    }

                    if show_about.get()
                    {

                        set_show_about
                            .set(false);

                        return;
                    }

                    if show_surprise
                        .get()
                    {

                        set_show_surprise.set(false);

                        return;
                    }
                }

                // Pagination with Arrows (only if no modal is open and not typing in an input)
                if !show_config.get()
                    && !show_about.get()
                    && !show_surprise
                        .get()
                {

                    let active_el = web_sys::window()
                     .and_then(|w| w.document())
                     .and_then(|d| d.active_element())
                     .map(|el| el.tag_name().to_uppercase())
                     .unwrap_or("BODY".to_string());

                    if active_el != "INPUT" && active_el != "TEXTAREA" && active_el != "SELECT" {
                     if key == "ArrowLeft" {
                        let current = page.get();
                        if current > 1 {
                             set_page.set(current - 1);
                             set_trigger_search.update(|s| s.page = current - 1);
                        }
                     } else if key == "ArrowRight" {
                         let current = page.get();
                         let total = total_count_resource.get()
                             .and_then(|res| res.ok())
                             .unwrap_or(0);
                         let p_size = page_size.get();
                         let max_page = total.div_ceil(p_size).max(1);

                         if current < max_page {
                             set_page.set(current + 1);
                             set_trigger_search.update(|s| s.page = current + 1);
                         }
                     }
                 }
                }
            },
        );
    }

    let on_page_change =
        move |p: usize| {

            set_page.set(p);

            set_trigger_search
                .update(|s| s.page = p);
            // Scroll to top of list
            // #[cfg(feature = "hydrate")]
            // {
            // if let Some(window) = web_sys::window() {
            // window.scroll_to_with_x_and_y(0.0, 400.0);
            // }
            // }
        };

    view! {
        <div class="max-w-7xl mx-auto px-4 py-8 md:px-8 md:py-12 space-y-12">
            <header class="flex flex-col md:flex-row md:items-end justify-between gap-6 border-b border-white/5 pb-10 animate-fade-in">
                <div class="space-y-2 animate-slide-up">
                    <div class="flex items-center gap-3">
                        <img src="/logo.svg" alt="arXiv Daily Logo" class="h-10 w-10 rounded-xl shadow-2xl shadow-[0_0_15px_rgba(59,130,246,0.5)]" />
                        <h1 class="text-5xl font-black text-obsidian-heading tracking-tighter">
                            "arXiv" <span class="text-obsidian-accent gradient-text text-glow">"Daily"</span>
                        </h1>
                    </div>
                    <p class="text-obsidian-text/40 font-medium ml-1 tracking-wide uppercase text-xs">"Personalized research discovery platform"</p>
                </div>
                <div class="flex items-center gap-3">
                    <div class="relative w-full md:w-96 group animate-slide-up stagger-1">
                        <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                            <svg class="h-5 w-5 text-obsidian-text/20 group-focus-within:text-obsidian-accent transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                        </div>
                        <input
                            type="text"
                            placeholder="Search research papers..."
                            class="w-full bg-obsidian-sidebar/50 backdrop-blur-xl border border-white/5 rounded-2xl pl-12 pr-4 py-3.5 focus:outline-none focus:ring-2 focus:ring-obsidian-accent/30 focus:border-obsidian-accent/30 transition-all text-obsidian-heading placeholder:text-obsidian-text/20 shadow-2xl"
                            on:input=move |ev| {
                                set_input_val.set(event_target_value(&ev));
                            }
                            on:keydown=move |ev| {
                                if ev.key() == "Enter" {
                                    on_search(());
                                }
                            }
                            prop:value=input_val
                        />
                    </div>
                    <div class="flex gap-2 animate-slide-up stagger-1">
                        <button
                            on:click=move |_| set_view_mode.set("card".to_string())
                            class=move || format!(
                                "p-3 rounded-xl transition-all border {}",
                                if view_mode.get() == "card" {
                                    "bg-obsidian-accent/20 border-obsidian-accent/30 text-obsidian-accent"
                                } else {
                                    "bg-obsidian-sidebar/50 border-white/5 text-obsidian-text/40 hover:text-obsidian-text hover:border-white/10"
                                }
                            )
                            title="Card View"
                        >
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                            </svg>
                        </button>
                        <button
                            on:click=move |_| set_view_mode.set("list".to_string())
                            class=move || format!(
                                "p-3 rounded-xl transition-all border {}",
                                if view_mode.get() == "list" {
                                    "bg-obsidian-accent/20 border-obsidian-accent/30 text-obsidian-accent"
                                } else {
                                    "bg-obsidian-sidebar/50 border-white/5 text-obsidian-text/40 hover:text-obsidian-text hover:border-white/10"
                                }
                            )
                            title="List View"
                        >
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                        </button>
                    </div>
                </div>
            </header>

            {move || {
                let pending = fetch_action.pending();
                let value = fetch_action.value();

                view! {
                    <Show when=move || show_fetch_status.get()>
                        <div class="fixed top-8 left-1/2 -translate-x-1/2 z-[100] animate-slide-up">
                            <div class="glass border border-white/10 rounded-2xl px-6 py-3 shadow-2xl flex items-center gap-4">
                                <Show
                                    when=move || pending.get()
                                    fallback=move || {
                                        value.with(|v| {
                                            match v {
                                                Some(Ok(count)) => {
                                                    let count = *count;
                                                    view! {
                                                        <div class="flex items-center gap-3">
                                                            <div class="w-2 h-2 bg-green-500 rounded-full shadow-[0_0_10px_#22c55e]"></div>
                                                            <p class="text-sm font-bold text-white tracking-tight">"Synchronized "{count}" new releases"</p>
                                                        </div>
                                                    }.into_any()
                                                }
                                                Some(Err(e)) => view! {
                                                    <div class="flex items-center gap-3">
                                                        <div class="w-2 h-2 bg-red-500 rounded-full shadow-[0_0_10px_#ef4444]"></div>
                                                        <p class="text-sm font-bold text-red-200">"Sync Error: "{e.to_string()}</p>
                                                    </div>
                                                }.into_any(),
                                                _ => ().into_any()
                                            }
                                        })
                                    }
                                >
                                    <div class="flex items-center gap-3">
                                        <div class="w-4 h-4 border-2 border-obsidian-accent/20 border-t-obsidian-accent rounded-full animate-spin"></div>
                                        <p class="text-sm font-bold text-obsidian-text tracking-tight animate-pulse">"Aggregating arXiv archive..."</p>
                                    </div>
                                </Show>
                            </div>
                        </div>
                    </Show>
                }
            }}

            <FilterBar
                selected_category=selected_category.into()
                set_selected_category
                date_filter=date_filter.into()
                set_date_filter
                on_fetch=Callback::new(move |_| {
                    fetch_action.dispatch((selected_category.get(), date_filter.get(), end_date_filter.get()));
                })
                fetch_pending=fetch_action.pending().into()
                on_search=Callback::new(on_search)
                on_reset=Callback::new(on_reset)
                on_edit_config=Callback::new(move |_| set_show_config.set(true))
                on_about=Callback::new(move |_| set_show_about.set(true))
                on_surprise=Callback::new(move |_| set_show_surprise.set(true))
                on_unload_model=Callback::new(move |_| { unload_action.dispatch(()); })
                end_date_filter=end_date_filter.into()
                set_end_date_filter
                negative_query=negative_query.into()
                set_negative_query
                use_llm=use_llm.into()
                set_use_llm=use_llm.write_only()
                page_size=page_size.into()
                set_page_size=page_size.write_only()
                view_mode=view_mode.into()
            />

            <ConfigModal
                show=show_config.into()
                on_close=Callback::new(move |_| set_show_config.set(false))
                use_llm=use_llm
                page_size=page_size
                fetch_action=fetch_action
                selected_category=selected_category.into()
            />
            <AboutModal show=show_about.into() on_close=Callback::new(move |_| set_show_about.set(false))/>
            <surprise::SurpriseModal show=show_surprise.into() on_close=Callback::new(move |_| set_show_surprise.set(false))/>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 gap-8">
                <Transition fallback=move || view! {
                    <div class="col-span-full py-48 flex flex-col items-center justify-center space-y-6 animate-fade-in">
                        <div class="relative">
                            <div class="absolute inset-0 bg-obsidian-accent/20 blur-3xl rounded-full animate-pulse"></div>
                            <div class="relative w-16 h-16 border-4 border-obsidian-accent/10 border-t-obsidian-accent rounded-full animate-spin"></div>
                        </div>
                        <p class="text-obsidian-text/30 font-semibold tracking-widest uppercase text-xs animate-pulse">"Accessing global knowledge base..."</p>
                    </div>
                }>
                    {move || {
                        match papers.get() {
                            Some(Ok(current_papers)) => {
                                if current_papers.is_empty() {
                                    view! {
                                        <div class="col-span-full py-32 text-center space-y-4">
                                            <div class="text-6xl text-obsidian-text/10 italic font-bold">"∅"</div>
                                            <p class="text-obsidian-text/40 text-lg">"No papers match your search criteria."</p>
                                        </div>
                                    }.into_any()
                                } else {
                                    let value = current_papers.clone();
                                    view! {
                                        <div class="col-span-full space-y-12">
                                            <div class=move || if view_mode.get() == "card" { "block" } else { "hidden" }>
                                                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 gap-8">
                                                    <For
                                                        each=move || value.clone()
                                                        key=|paper| paper.id.clone()
                                                        children=move |paper| {
                                                            let paper_id = paper.id.clone();
                                                            let is_hidden = move || hidden_paper_ids.get().contains(&paper_id);
                                                            view! {
                                                                <Show when=move || !is_hidden()>
                                                                    <PaperCard paper=paper.clone()/>
                                                                </Show>
                                                            }
                                                        }
                                                    />
                                                </div>
                                            </div>
                                            <div class=move || if view_mode.get() == "list" { "block space-y-4" } else { "hidden" }>
                                                <For
                                                    each=move || current_papers.clone()
                                                    key=|paper| paper.id.clone()
                                                    children=move |paper| {
                                                        let paper_id = paper.id.clone();
                                                        let is_hidden = move || hidden_paper_ids.get().contains(&paper_id);
                                                        view! {
                                                            <Show when=move || !is_hidden()>
                                                                <PaperListItem paper=paper.clone()/>
                                                            </Show>
                                                        }
                                                    }
                                                />
                                            </div>
                                             <Pagination
                                                current_page=page.into()
                                                total_count=total_count
                                                page_size=page_size.get()
                                                on_page_change=Callback::new(on_page_change)
                                            />
                                        </div>
                                    }.into_any()
                                }
                            }
                            Some(Err(e)) => {
                                view! {
                                    <div class="col-span-full py-20 bg-red-500/10 border border-red-500/20 rounded-2xl p-8 text-center space-y-4">
                                        <div class="text-red-400 text-4xl">"⚠"</div>
                                        <h3 class="text-xl font-bold text-red-200">"Data Fetching Error"</h3>
                                        <p class="text-red-200/60 font-mono text-sm">{e.to_string()}</p>
                                        <button
                                            on:click=move |_| papers.refetch()
                                            class="px-6 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-200 rounded-lg transition-colors border border-red-500/30"
                                        >
                                            "Try Again"
                                        </button>
                                    </div>
                                }.into_any()
                            }
                            None => {
                                ().into_any()
                            }
                        }
                    }}
                </Transition>
            </div>

            <BackToTop/>

            <footer class="pt-20 pb-10 border-t border-white/5 text-center">
                <p class="text-xs text-obsidian-text/20 uppercase tracking-[0.2em] font-bold">
                    "Made by Apich Organization"
                </p>
            </footer>
        </div>
    }
}

#[component]

fn PaperCard(
    paper: Paper
) -> impl IntoView {

    let (expanded, set_expanded) =
        signal(false);

    let authors = paper.authors_list();

    let display_authors =
        authors.join(", ");

    let title = paper.title.clone();

    let url = paper.url.clone();

    let summary = paper
        .summary
        .clone();

    let pdf_link = paper
        .pdf_link
        .clone();

    let published_str = paper
        .published_date()
        .format("%b %d, %Y")
        .to_string();

    let category_name =
        paper.primary_category_name();

    view! {
        <div class="glass glow-hover rounded-3xl p-8 flex flex-col h-full active:scale-[0.98] animate-slide-up group relative overflow-hidden">
            <div class="absolute -top-12 -right-12 w-24 h-24 bg-obsidian-accent/10 blur-[60px] rounded-full group-hover:bg-obsidian-accent/20 transition-colors"></div>

            <div class="flex items-center justify-between gap-4 mb-6 relative z-10">
                <span class="inline-flex items-center px-3 py-1.5 rounded-full text-[10px] font-black bg-obsidian-accent/5 text-obsidian-accent border border-obsidian-accent/10 uppercase tracking-[0.1em] backdrop-blur-sm">
                    {category_name}
                </span>
                <span class="text-[10px] uppercase tracking-[0.15em] text-obsidian-text/30 font-bold">
                    {published_str}
                </span>
            </div>

            <h3 class="text-xl font-bold text-obsidian-heading leading-[1.3] group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-obsidian-accent group-hover:to-obsidian-accent-light transition-all duration-500 mb-3 relative z-10 font-sans">
                <a
                    href=url.clone()
                    target="_blank"
                    class="hover:underline decoration-obsidian-accent/20 decoration-2 underline-offset-4 transition-all"
                    inner_html=move || render_math(&title)
                ></a>
            </h3>

            <p class="text-[11px] text-obsidian-text/30 font-semibold mb-6 line-clamp-1 italic tracking-wide group-hover:text-obsidian-text/50 transition-colors">
                {display_authors}
            </p>

            <div class="relative flex-grow z-10">
                <div
                    class=move || format!("text-[13px] leading-relaxed text-obsidian-text/60 transition-all duration-700 {}", if expanded.get() { "opacity-100" } else { "line-clamp-4 overflow-hidden opacity-80" })
                    inner_html={
                        let value = summary.clone();
                        move || render_math(&value)
                    }
                ></div>
                {move || if !expanded.get() && summary.len() > 180 {
                    view! {
                        <button
                            on:click=move |_| set_expanded.set(true)
                            class="text-[10px] font-black text-obsidian-accent hover:text-obsidian-accent-light mt-4 flex items-center gap-2 group/btn transition-colors focus:outline-none uppercase tracking-widest"
                        >
                            "View Details"
                            <svg class="w-3.5 h-3.5 group-hover/btn:translate-x-1.5 transition-transform cubic-bezier(0.34, 1.56, 0.64, 1)" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                            </svg>
                        </button>
                    }.into_any()
                } else {
                    ().into_any()
                }}
            </div>

            <div class="mt-8 flex items-center gap-4 relative z-10">
                {if let Some(link) = pdf_link {
                    view! {
                        <a
                            href=link
                            target="_blank"
                            class="flex-1 inline-flex items-center justify-center px-6 py-3.5 text-xs font-black text-white bg-gradient-to-br from-obsidian-accent to-obsidian-accent/80 hover:from-obsidian-accent-light hover:to-obsidian-accent rounded-2xl transition-all shadow-[0_10px_25px_-10px_rgba(59,130,246,0.5)] uppercase tracking-widest active:scale-[0.96]"
                        >
                            "Access PDF"
                            <svg class="w-4 h-4 ml-2.5 opacity-80" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                        </a>
                    }.into_any()
                } else {
                    view! {
                         <div class="flex-1 py-3.5 text-center text-[10px] text-obsidian-text/20 uppercase tracking-[0.2em] font-black border border-white/5 rounded-2xl">"Paper Locked"</div>
                    }.into_any()
                }}

                <a
                    href=url
                    target="_blank"
                    class="p-3 text-obsidian-text/40 hover:text-obsidian-accent bg-white/5 hover:bg-white/10 rounded-xl transition-all"
                    title="View on arXiv"
                >
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                </a>
            </div>
        </div>
    }
}

#[component]

fn ConfigModal(
    show: Signal<bool>,
    on_close: Callback<()>,
    use_llm: RwSignal<bool>,
    page_size: RwSignal<usize>,
    fetch_action: Action<
        (
            String,
            String,
            String,
        ),
        Result<usize, ServerFnError>,
    >,
    selected_category: Signal<String>,
) -> impl IntoView {

    let (is_admin, set_is_admin) =
        signal(false);

    let (
        admin_password,
        set_admin_password,
    ) = signal("".to_string());

    let (auth_error, set_auth_error) =
        signal(None::<String>);

    let (
        show_password_change,
        set_show_password_change,
    ) = signal(false);

    let (
        new_password,
        set_new_password,
    ) = signal("".to_string());

    let (
        confirm_password,
        set_confirm_password,
    ) = signal("".to_string());

    let config_resource = Resource::new(
        move || {

            (
                show.get(),
                is_admin.get(),
            )
        },
        |(show, is_admin)| {

            async move {

                if show && is_admin {

                    get_config().await
                } else {

                    Ok("".to_string())
                }
            }
        },
    );

    let arxiv_resource = Resource::new(
        move || {

            (
                show.get(),
                is_admin.get(),
            )
        },
        |(show, is_admin)| {

            async move {

                if show && is_admin {

                    get_arxiv_config()
                        .await
                } else {

                    Ok(ArxivConfig::default())
                }
            }
        },
    );

    let (arxiv_cat, set_arxiv_cat) =
        signal("".to_string());

    let (arxiv_start, set_arxiv_start) =
        signal(0);

    let (arxiv_max, set_arxiv_max) =
        signal(50);

    let (
        arxiv_lookback,
        set_arxiv_lookback,
    ) = signal(7);

    let (sync_start, set_sync_start) =
        signal("".to_string());

    let (sync_end, set_sync_end) =
        signal(
            chrono::Utc::now()
                .format("%Y-%m-%d")
                .to_string(),
        );

    Effect::new(move |_| {
        if let Some(Ok(c)) =
            arxiv_resource.get()
        {

            set_arxiv_cat
                .set(c.category);

            set_arxiv_start
                .set(c.start);

            set_arxiv_max
                .set(c.max_results);

            set_arxiv_lookback
                .set(c.lookback_days);
        }
    });

    let (
        manual_cat_override,
        set_manual_cat_override,
    ) = signal(None::<String>);

    let save_arxiv_action = Action::new(
        |input: &(
            String,
            ArxivConfig,
        )| {

            let password =
                input.0.clone();

            let arxiv = input.1.clone();

            async move {

                save_arxiv_config(
                    password,
                    arxiv,
                )
                .await
            }
        },
    );

    let (
        local_fetching,
        set_local_fetching,
    ) = signal(false);

    // Load local settings
    Effect::new(move |_| {

        #[cfg(feature = "hydrate")]
        {

            if let Some(window) =
                web_sys::window()
            {

                if let Ok(Some(
                    storage,
                )) = window
                    .local_storage()
                {

                    if let Ok(Some(val)) =
                        storage.get("local_fetching")
                    {
                        set_local_fetching
                            .set(val == "true");
                    }
                }
            }
        }
    });

    let save_config_action =
        Action::new(
            |input: &(
                String,
                String,
            )| {

                let password =
                    input.0.clone();

                let content =
                    input.1.clone();

                async move {

                    save_config(
                        password,
                        content,
                    )
                    .await
                }
            },
        );

    let update_password_action =
        Action::new(
            |input: &(
                String,
                String,
            )| {

                let old =
                    input.0.clone();

                let new =
                    input.1.clone();

                async move {

                    update_admin_password(old, new).await
                }
            },
        );

    let (
        config_content,
        set_config_content,
    ) = signal("".to_string());

    Effect::new(move |_| {
        if let Some(Ok(c)) =
            config_resource.get()
        {

            set_config_content.set(c);
        }
    });

    let verify_admin_action =
        Action::new(
            |password: &String| {

                let p =
                    password.clone();

                async move {

                    verify_admin_password(p).await
                }
            },
        );

    Effect::new(move |_| {
        if let Some(Ok(true)) =
            verify_admin_action
                .value()
                .get()
        {

            set_is_admin.set(true);

            set_auth_error.set(None);
        } else if let Some(Ok(false)) =
            verify_admin_action
                .value()
                .get()
        {

            set_auth_error.set(Some(
                "Invalid admin \
                 password"
                    .to_string(),
            ));
        }
    });

    view! {
        <Show when=move || show.get()>
            <div class="fixed inset-0 z-[100] flex items-center justify-center p-4 md:p-6 bg-black/60 backdrop-blur-xl animate-fade-in">
                <div class="glass-dark border border-white/10 rounded-[2.5rem] w-full max-w-2xl shadow-[0_50px_100px_-20px_rgba(0,0,0,0.5)] overflow-hidden animate-scale-in">
                    // Header
                    <div class="px-8 py-6 border-b border-white/5 flex justify-between items-center bg-white/5">
                        <div class="flex items-center gap-4">
                            <div class="p-2.5 bg-obsidian-accent/10 rounded-2xl">
                                <svg class="w-6 h-6 text-obsidian-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                </svg>
                            </div>
                            <div>
                                <h2 class="text-xl font-black text-obsidian-heading tracking-tight">"Configuration"</h2>
                                <p class="text-[10px] text-obsidian-text/40 font-bold uppercase tracking-widest">
                                    {move || if is_admin.get() { "Administrator Access" } else { "User Preferences" }}
                                </p>
                            </div>
                        </div>
                        <button on:click=move |_| {
                            on_close.run(());
                            set_is_admin.set(false);
                            set_admin_password.set("".to_string());
                        } class="w-10 h-10 rounded-full flex items-center justify-center text-obsidian-text/40 hover:bg-white/5 hover:text-white transition-all bg-white/5 border border-white/5">"✕"</button>
                    </div>

                    // Content
                    <div class="p-8 space-y-8 max-h-[70vh] overflow-y-auto custom-scrollbar">
                        <Show
                            when=move || is_admin.get()
                            fallback=move || view! {
                                <div class="space-y-8 animate-fade-in">
                                    // User Settings Section
                                    <div class="space-y-6">
                                        <div class="flex items-center justify-between p-6 bg-white/5 rounded-3xl border border-white/5 hover:border-obsidian-accent/20 transition-all group">
                                            <div class="space-y-1">
                                                <h3 class="font-bold text-obsidian-heading">"AI Assistance"</h3>
                                                <p class="text-xs text-obsidian-text/40">"Toggle LLM-powered paper filtering and enrichment"</p>
                                            </div>
                                            <button
                                                on:click=move |_| use_llm.update(|v| *v = !*v)
                                                class=move || format!(
                                                    "w-12 h-6 rounded-full transition-all flex items-center px-1 {}",
                                                    if use_llm.get() { "bg-obsidian-accent" } else { "bg-white/10" }
                                                )
                                            >
                                                <div class=move || format!(
                                                    "w-4 h-4 bg-white rounded-full shadow-lg transition-all transform {}",
                                                    if use_llm.get() { "translate-x-6" } else { "translate-x-0" }
                                                ) />
                                            </button>
                                        </div>

                                        <div class="flex items-center justify-between p-6 bg-white/5 rounded-3xl border border-white/5 hover:border-obsidian-accent/20 transition-all group">
                                            <div class="space-y-1">
                                                <h3 class="font-bold text-obsidian-heading">"Local Fetching"</h3>
                                                <p class="text-xs text-obsidian-text/40">"Perform RSS fetching directly from browser"</p>
                                            </div>
                                            <button
                                                on:click=move |_| {
                                                    let new_val = !local_fetching.get();
                                                    set_local_fetching.set(new_val);
                                                    #[cfg(feature = "hydrate")]
                                                    {
                                                        if let Some(window) = web_sys::window() {
                                                            if let Ok(Some(storage)) = window.local_storage() {
                                                                let _ = storage.set("local_fetching", if new_val { "true" } else { "false" });
                                                            }
                                                        }
                                                    }
                                                }
                                                class=move || format!(
                                                    "w-12 h-6 rounded-full transition-all flex items-center px-1 {}",
                                                    if local_fetching.get() { "bg-obsidian-accent" } else { "bg-white/10" }
                                                )
                                            >
                                                <div class=move || format!(
                                                    "w-4 h-4 bg-white rounded-full shadow-lg transition-all transform {}",
                                                    if local_fetching.get() { "translate-x-6" } else { "translate-x-0" }
                                                ) />
                                            </button>
                                        </div>

                                        <div class="p-6 bg-white/5 rounded-3xl border border-white/5 space-y-4">
                                            <div class="px-2 flex items-center justify-between mb-2">
                                                <div class="space-y-1">
                                                    <h3 class="font-bold text-obsidian-heading">"Manual Archive Sync"</h3>
                                                    <p class="text-[10px] text-obsidian-text/30 uppercase tracking-widest font-black">"Sync specific date range"</p>
                                                </div>
                                                <div class="w-1.5 h-1.5 rounded-full bg-obsidian-accent animate-pulse" />
                                            </div>

                                            <div class="flex items-center gap-3 px-2 mb-4">
                                                <button
                                                    on:click=move |_| {
                                                        let now = chrono::Utc::now();
                                                        set_sync_end.set(now.format("%Y-%m-%d").to_string());
                                                        set_sync_start.set((now - chrono::TimeDelta::try_days(1).unwrap_or_default()).format("%Y-%m-%d").to_string());
                                                    }
                                                    class="text-[8px] font-black uppercase tracking-widest text-obsidian-text/30 hover:text-obsidian-accent transition-colors"
                                                >
                                                    "Last 24h"
                                                </button>
                                                <span class="text-obsidian-text/10 text-[8px]">"•"</span>
                                                <button
                                                    on:click=move |_| {
                                                        let now = chrono::Utc::now();
                                                        set_sync_end.set(now.format("%Y-%m-%d").to_string());
                                                        set_sync_start.set((now - chrono::TimeDelta::try_days(7).unwrap_or_default()).format("%Y-%m-%d").to_string());
                                                    }
                                                    class="text-[8px] font-black uppercase tracking-widest text-obsidian-text/30 hover:text-obsidian-accent transition-colors"
                                                >
                                                    "Last 7d"
                                                </button>
                                                <span class="text-obsidian-text/10 text-[8px]">"•"</span>
                                                <button
                                                    on:click=move |_| {
                                                        let now = chrono::Utc::now();
                                                        set_sync_end.set(now.format("%Y-%m-%d").to_string());
                                                        set_sync_start.set((now - chrono::TimeDelta::try_days(30).unwrap_or_default()).format("%Y-%m-%d").to_string());
                                                    }
                                                    class="text-[8px] font-black uppercase tracking-widest text-obsidian-text/30 hover:text-obsidian-accent transition-colors"
                                                >
                                                    "Last 30d"
                                                </button>
                                            </div>

                                            <div class="grid grid-cols-2 gap-3">
                                                <div class="space-y-2">
                                                    <label class="text-[9px] font-black text-obsidian-text/20 uppercase tracking-widest ml-2">"Start Date"</label>
                                                    <input
                                                        type="date"
                                                        class="w-full bg-black/40 border border-white/5 rounded-2xl px-4 py-3 text-xs focus:outline-none focus:ring-2 focus:ring-obsidian-accent/20 transition-all font-mono text-obsidian-text/80"
                                                        on:input=move |ev| set_sync_start.set(event_target_value(&ev))
                                                        prop:value=sync_start
                                                    />
                                                </div>
                                                <div class="space-y-2">
                                                    <label class="text-[9px] font-black text-obsidian-text/20 uppercase tracking-widest ml-2">"End Date"</label>
                                                    <input
                                                        type="date"
                                                        class="w-full bg-black/40 border border-white/5 rounded-2xl px-4 py-3 text-xs focus:outline-none focus:ring-2 focus:ring-obsidian-accent/20 transition-all font-mono text-obsidian-text/80"
                                                        on:input=move |ev| set_sync_end.set(event_target_value(&ev))
                                                        prop:value=sync_end
                                                    />
                                                </div>
                                            </div>

                                            <button
                                                on:click=move |_| {
                                                    fetch_action.dispatch((
                                                        selected_category.get(),
                                                        sync_start.get(),
                                                        sync_end.get()
                                                    ));
                                                }
                                                class="w-full py-4 bg-obsidian-accent/10 hover:bg-obsidian-accent/20 text-obsidian-accent text-[9px] font-black uppercase tracking-widest rounded-2xl transition-all border border-obsidian-accent/20 flex items-center justify-center gap-2 group"
                                                disabled=move || fetch_action.pending().get()
                                            >
                                                {move || if fetch_action.pending().get() {
                                                    view! { <div class="w-3 h-3 border-2 border-obsidian-accent/20 border-t-obsidian-accent rounded-full animate-spin"></div> }.into_any()
                                                } else {
                                                    view! {
                                                        <svg class="w-3.5 h-3.5 group-hover:rotate-180 transition-transform duration-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                                        </svg>
                                                    }.into_any()
                                                }}
                                                "Trigger Fetch Cycle"
                                            </button>
                                        </div>

                                        <div class="p-6 bg-white/5 rounded-3xl border border-white/5 space-y-4">
                                            <div class="flex justify-between items-center">
                                                <h3 class="font-bold text-obsidian-heading">"Default Page Size"</h3>
                                                <span class="text-xs font-black text-obsidian-accent bg-obsidian-accent/10 px-3 py-1 rounded-full">{move || page_size.get()} " items"</span>
                                            </div>
                                            <input
                                                type="range"
                                                min="10"
                                                max="100"
                                                step="5"
                                                prop:value=move || page_size.get().to_string()
                                                on:input=move |ev| {
                                                    if let Ok(val) = event_target_value(&ev).parse::<usize>() {
                                                        page_size.set(val);
                                                    }
                                                }
                                                class="w-full accent-obsidian-accent opacity-70 hover:opacity-100 transition-opacity cursor-pointer"
                                            />
                                        </div>
                                    </div>

                                    // Switch to Admin Button
                                    <div class="pt-8 border-t border-white/5 flex flex-col items-center gap-4">
                                        <p class="text-[10px] font-black text-obsidian-text/20 uppercase tracking-[0.2em]">"Administrative Controls"</p>
                                        <div class="flex w-full gap-2">
                                            <input
                                                type="password"
                                                placeholder="Enter Admin Password"
                                                class="flex-1 bg-white/5 border border-white/5 rounded-2xl px-5 py-3.5 text-sm focus:outline-none focus:ring-2 focus:ring-obsidian-accent/20 transition-all font-mono"
                                                on:input=move |ev| set_admin_password.set(event_target_value(&ev))
                                                on:keydown=move |ev| {
                                                    if ev.key() == "Enter" {
                                                        verify_admin_action.dispatch(admin_password.get());
                                                    }
                                                }
                                                prop:value=admin_password
                                            />
                                            <button
                                                on:click=move |_| { verify_admin_action.dispatch(admin_password.get()); }
                                                class="px-6 bg-white/10 hover:bg-white/20 text-white text-xs font-black uppercase tracking-widest rounded-2xl transition-all"
                                            >
                                                "Login"
                                            </button>
                                        </div>
                                        <Show when=move || auth_error.get().is_some()>
                                            <p class="text-red-400 text-[10px] font-bold uppercase tracking-wider">{move || auth_error.get()}</p>
                                        </Show>
                                    </div>
                                </div>
                            }
                        >
                            // Admin Panel view content
                            <div class="space-y-8 animate-fade-in">
                                <Show
                                    when=move || !show_password_change.get()
                                    fallback=move || view! {
                                        <div class="space-y-6">
                                            <div class="flex items-center gap-3 mb-2">
                                                <button on:click=move |_| set_show_password_change.set(false) class="p-2 hover:bg-white/5 rounded-xl transition-all text-obsidian-text/40">"←"</button>
                                                <h3 class="font-bold text-obsidian-heading">"Change Admin Password"</h3>
                                            </div>
                                            <div class="space-y-4">
                                                <div class="space-y-2">
                                                    <label class="text-[10px] font-black text-obsidian-text/20 uppercase tracking-widest ml-2">"New Password"</label>
                                                    <input
                                                        type="password"
                                                        class="w-full bg-white/5 border border-white/10 rounded-2xl px-5 py-4 text-sm focus:outline-none focus:ring-2 focus:ring-obsidian-accent/20 transition-all font-mono"
                                                        on:input=move |ev| set_new_password.set(event_target_value(&ev))
                                                        prop:value=new_password
                                                    />
                                                </div>
                                                <div class="space-y-2">
                                                    <label class="text-[10px] font-black text-obsidian-text/20 uppercase tracking-widest ml-2">"Confirm Password"</label>
                                                    <input
                                                        type="password"
                                                        class="w-full bg-white/5 border border-white/10 rounded-2xl px-5 py-4 text-sm focus:outline-none focus:ring-2 focus:ring-obsidian-accent/20 transition-all font-mono"
                                                        on:input=move |ev| set_confirm_password.set(event_target_value(&ev))
                                                        prop:value=confirm_password
                                                    />
                                                </div>
                                                <button
                                                    on:click=move |_| {
                                                        if new_password.get() == confirm_password.get() && !new_password.get().is_empty() {
                                                            update_password_action.dispatch((admin_password.get(), new_password.get()));
                                                            set_show_password_change.set(false);
                                                            set_new_password.set("".to_string());
                                                            set_confirm_password.set("".to_string());
                                                        }
                                                    }
                                                    class="w-full py-4 bg-obsidian-accent text-white text-xs font-black uppercase tracking-widest rounded-2xl hover:brightness-110 transition-all"
                                                >
                                                    "Update Password"
                                                </button>
                                            </div>
                                        </div>
                                    }
                                >
                                    <div class="space-y-8 animate-fade-in">
                                        // Structured Config Section
                                        <div class="space-y-6 bg-white/5 p-6 rounded-[2rem] border border-white/5">
                                            <div class="flex items-center justify-between px-2">
                                                <label class="text-[10px] font-black text-obsidian-text/30 uppercase tracking-[0.2em]">"Quick Arxiv Config"</label>
                                                <Show when=move || save_arxiv_action.value().get().is_some()>
                                                    <span class="text-[9px] font-black text-green-400 uppercase tracking-widest animate-pulse">"✓ Saved"</span>
                                                </Show>
                                            </div>
                                            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                                                <div class="space-y-2">
                                                    <label class="text-[9px] font-black text-obsidian-text/20 uppercase tracking-widest ml-2">"Category"</label>
                                                    <input
                                                        type="text"
                                                        class="w-full bg-black/20 border border-white/5 rounded-2xl px-5 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-obsidian-accent/20 transition-all font-mono"
                                                        on:input=move |ev| set_arxiv_cat.set(event_target_value(&ev))
                                                        prop:value=arxiv_cat
                                                    />
                                                </div>
                                                <div class="space-y-2">
                                                    <label class="text-[9px] font-black text-obsidian-text/20 uppercase tracking-widest ml-2">"Start Offset"</label>
                                                    <input
                                                        type="number"
                                                        class="w-full bg-black/20 border border-white/5 rounded-2xl px-5 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-obsidian-accent/20 transition-all font-mono"
                                                        on:input=move |ev| {
                                                            if let Ok(val) = event_target_value(&ev).parse::<i32>() {
                                                                set_arxiv_start.set(val);
                                                            }
                                                        }
                                                        prop:value=arxiv_start
                                                    />
                                                </div>
                                                <div class="space-y-2">
                                                    <label class="text-[9px] font-black text-obsidian-text/20 uppercase tracking-widest ml-2">"Max Results"</label>
                                                    <input
                                                        type="number"
                                                        class="w-full bg-black/20 border border-white/5 rounded-2xl px-5 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-obsidian-accent/20 transition-all font-mono"
                                                        on:input=move |ev| {
                                                            if let Ok(val) = event_target_value(&ev).parse::<i32>() {
                                                                set_arxiv_max.set(val);
                                                            }
                                                        }
                                                        prop:value=arxiv_max
                                                    />
                                                </div>
                                                <div class="space-y-2">
                                                    <div class="flex justify-between px-2">
                                                        <label class="text-[9px] font-black text-obsidian-text/20 uppercase tracking-widest">"Lookback Window"</label>
                                                        <span class="text-[9px] font-black text-obsidian-accent">{move || arxiv_lookback.get()} " d"</span>
                                                    </div>
                                                    <div class="flex items-center gap-3">
                                                        <input
                                                            type="range"
                                                            min="1"
                                                            max="90"
                                                            prop:value=move || arxiv_lookback.get().to_string()
                                                            on:input=move |ev| {
                                                                if let Ok(val) = event_target_value(&ev).parse::<i32>() {
                                                                    set_arxiv_lookback.set(val);
                                                                }
                                                            }
                                                            class="flex-1 h-1 bg-black/40 rounded-lg appearance-none cursor-pointer accent-obsidian-accent"
                                                        />
                                                    </div>
                                                </div>
                                            </div>
                                            <div class="flex justify-end">
                                                <button
                                                    on:click=move |_| {
                                                        save_arxiv_action.dispatch((admin_password.get(), ArxivConfig {
                                                            category: arxiv_cat.get(),
                                                            start: arxiv_start.get(),
                                                            max_results: arxiv_max.get(),
                                                            lookback_days: arxiv_lookback.get(),
                                                        }));
                                                    }
                                                    class="px-6 py-2.5 bg-obsidian-accent/10 hover:bg-obsidian-accent/20 text-obsidian-accent text-[9px] font-black uppercase tracking-widest rounded-xl transition-all border border-obsidian-accent/20"
                                                >
                                                    "Update Arxiv Defaults"
                                                </button>
                                            </div>
                                        </div>

                                        <div class="space-y-4">
                                        <div class="flex justify-between items-end px-2">
                                            <label class="text-[10px] font-black text-obsidian-text/30 uppercase tracking-[0.2em]">"Global TOML Config"</label>
                                            <button
                                                on:click=move |_| set_show_password_change.set(true)
                                                class="text-[10px] font-black text-obsidian-accent/60 hover:text-obsidian-accent uppercase tracking-widest transition-colors"
                                            >
                                                "Manage Security"
                                            </button>
                                        </div>
                                        <textarea
                                            class="w-full h-80 bg-black/40 border border-white/5 rounded-[2rem] p-6 font-mono text-xs text-obsidian-text/80 focus:outline-none focus:ring-2 focus:ring-obsidian-accent/20 transition-all custom-scrollbar outline-none shadow-inner"
                                            on:input=move |ev| set_config_content.set(event_target_value(&ev))
                                            prop:value=config_content
                                        />
                                        <div class="flex justify-end gap-3 pt-4">
                                            <button
                                                on:click=move |_| set_is_admin.set(false)
                                                class="px-6 py-3 text-[10px] font-black uppercase tracking-widest text-obsidian-text/40 hover:text-white"
                                            >
                                                "Logout"
                                            </button>
                                            <button
                                                on:click=move |_| {
                                                    save_config_action.dispatch((admin_password.get(), config_content.get()));
                                                }
                                                class="px-8 py-3 bg-obsidian-accent text-white text-[10px] font-black uppercase tracking-widest rounded-xl hover:brightness-110 transition-all shadow-lg"
                                            >
                                                "Persist to Disk"
                                            </button>
                                        </div>
                                    </div>
                                    </div>
                                </Show>
                            </div>
                        </Show>
                    </div>

                    // Footer message
                    <div class="px-8 py-5 bg-black/20 border-t border-white/5 flex items-center justify-center gap-3">
                        <div class="w-1.5 h-1.5 rounded-full bg-obsidian-accent animate-pulse" />
                        <p class="text-[9px] font-bold text-obsidian-text/30 uppercase tracking-[0.15em]">
                            "Changes may require a system refresh to take effect"
                        </p>
                    </div>
                </div>
            </div>
        </Show>
    }
}

#[component]

fn Pagination(
    current_page: Signal<usize>,
    total_count: Resource<
        Result<usize, ServerFnError>,
    >,
    page_size: usize,
    on_page_change: Callback<usize>,
) -> impl IntoView {

    let total_pages =
        Memo::new(move |_| {

            total_count
                .get()
                .and_then(|res| {

                    res.ok()
                })
                .map(|c| {

                    c.div_ceil(
                        page_size,
                    )
                })
                .unwrap_or(1)
        });

    let can_go_prev =
        move || current_page.get() > 1;

    let can_go_next = move || {

        current_page.get()
            < total_pages.get()
    };

    view! {
        <div class="flex flex-col items-center gap-6 pt-12 border-t border-white/5">
            <div class="flex items-center gap-4">
                <button
                    on:click=move |_| if can_go_prev() { on_page_change.run(current_page.get() - 1) }
                    disabled=move || !can_go_prev()
                    class="p-3 rounded-2xl glass hover:bg-white/5 disabled:opacity-20 transition-all text-obsidian-text active:scale-90"
                >
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M15 19l-7-7 7-7" />
                    </svg>
                </button>

                <div class="flex items-center gap-4 px-6 py-2 glass rounded-2xl shadow-inner">
                    <span class="text-[10px] font-black uppercase tracking-widest text-obsidian-text/30">"PAGE"</span>
                    <input
                        type="number"
                        min="1"
                        max=move || total_pages.get()
                        prop:value=move || current_page.get()
                        class="w-16 bg-transparent border-none p-0 text-center text-lg font-black text-obsidian-accent focus:outline-none"
                        on:input=move |ev| {
                            let val = event_target_value(&ev);
                            if let Ok(n) = val.parse::<usize>()
                                && n >= 1 && n <= total_pages.get() {
                                    on_page_change.run(n);
                                }
                            }
                        on:keydown=move |ev| {
                            if ev.key() == "Enter" {
                                let val = event_target_value(&ev);
                                if let Ok(n) = val.parse::<usize>()
                                    && n >= 1 && n <= total_pages.get() {
                                        on_page_change.run(n);
                                    }

                            }
                        }
                    />
                    <span class="text-[10px] font-black uppercase tracking-widest text-obsidian-text/30">"OF"</span>
                    <span class="text-lg font-black text-obsidian-text/50">{move || total_pages.get()}</span>
                </div>

                <button
                    on:click=move |_| if can_go_next() { on_page_change.run(current_page.get() + 1) }
                    disabled=move || !can_go_next()
                    class="p-3 rounded-2xl glass hover:bg-white/5 disabled:opacity-20 transition-all text-obsidian-text active:scale-90"
                >
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M9 5l7 7-7 7" />
                    </svg>
                </button>
            </div>

            <div class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-text/20">
                {move || total_count.get().and_then(|res| res.ok()).unwrap_or(0)} " total papers discovered"
            </div>
        </div>
    }
}

#[component]

fn FilterBar(
    selected_category: Signal<String>,
    set_selected_category: WriteSignal<
        String,
    >,
    date_filter: Signal<String>,
    set_date_filter: WriteSignal<
        String,
    >,
    end_date_filter: Signal<String>,
    set_end_date_filter: WriteSignal<
        String,
    >,
    negative_query: Signal<String>,
    set_negative_query: WriteSignal<
        String,
    >,
    on_fetch: Callback<()>,
    fetch_pending: Signal<bool>,
    on_search: Callback<()>,
    on_reset: Callback<()>,
    on_edit_config: Callback<()>,
    on_about: Callback<()>,
    on_surprise: Callback<()>,
    on_unload_model: Callback<()>,
    use_llm: Signal<bool>,
    set_use_llm: WriteSignal<bool>,
    page_size: Signal<usize>,
    set_page_size: WriteSignal<usize>,
    view_mode: Signal<String>,
) -> impl IntoView {

    let (
        category_search,
        set_category_search,
    ) = signal("".to_string());

    let (is_open, set_is_open) =
        signal(false);

    let filtered_categories = Memo::new(
        move |_| {

            let search =
                category_search
                    .get()
                    .to_lowercase();

            let mut base = vec![(
                "all",
                "All Categories",
            )];

            base.extend(Category::ALL_CATEGORIES.iter().map(|(c, n)| (*c, *n)));

            if search.is_empty() {

                base
            } else {

                base.into_iter()
                .filter(|(code, name)| {
                    code.to_lowercase().contains(&search) || name.to_lowercase().contains(&search)
                })
                .collect()
            }
        },
    );

    let current_category_name =
        move || {

            let code =
                selected_category.get();

            if code == "all" {

                "All Categories"
            } else {

                Category::ALL_CATEGORIES
                    .iter()
                    .find(|(c, _)| {

                        *c == code
                    })
                    .map(|(_, n)| *n)
                    .unwrap_or(
                        "Unknown",
                    )
            }
        };

    view! {
        <div class="glass flex flex-wrap items-center gap-6 p-6 rounded-[2rem] border border-white/5 shadow-2xl animate-fade-in stagger-2 relative z-50">
            <div class="flex flex-col gap-2 min-w-[280px] flex-2 relative">
                <label class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-text/20 ml-2">"Research Category"</label>

                    <button
                        on:click=move |_| set_is_open.update(|v| *v = !*v)
                        class="w-full bg-black/20 backdrop-blur-md border border-white/5 rounded-2xl px-5 py-3 text-sm text-obsidian-heading flex items-center justify-between hover:border-obsidian-accent/30 transition-all focus:outline-none group shadow-inner"
                    >
                        <span class="truncate font-semibold tracking-tight">{current_category_name}</span>
                        <svg class=move || format!("w-4 h-4 text-obsidian-text/30 transition-transform duration-500 cubic-bezier(0.34, 1.56, 0.64, 1) {}", if is_open.get() { "rotate-180 text-obsidian-accent" } else { "" }) fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M19 9l-7 7-7-7" />
                        </svg>
                    </button>

                    <Show when=move || is_open.get()>
                        <div class="absolute z-[999] mt-3 w-full bg-obsidian-card border border-white/10 shadow-[0_30px_60px_-15px_rgba(0,0,0,0.8)] rounded-[1.5rem] overflow-hidden animate-scale-in">
                            <div class="p-2 border-b border-white/5">
                                <input
                                    type="text"
                                    placeholder="Filter categories..."
                                    class="w-full bg-obsidian-bg border border-white/5 rounded-lg px-3 py-2 text-xs text-obsidian-heading focus:outline-none focus:ring-1 focus:ring-obsidian-accent/40 placeholder:text-obsidian-text/20"
                                    on:input=move |ev| set_category_search.set(event_target_value(&ev))
                                    on:click=move |ev| ev.stop_propagation()
                                    on:keydown=move |ev| {
                                        if ev.key() == "Enter" {
                                            on_search.run(());
                                            set_is_open.set(false);
                                        }
                                    }
                                    prop:value=category_search
                                />
                            </div>
                            <div class="max-h-60 overflow-y-auto custom-scrollbar">
                                <For
                                    each=move || filtered_categories.get()
                                    key=|(code, _)| code.to_string()
                                    children=move |(code, name)| {
                                        let is_active = move || selected_category.get() == code;
                                        view! {
                                            <button
                                                on:click=move |_| {
                                                    set_selected_category.set(code.to_string());
                                                    set_is_open.set(false);
                                                    set_category_search.set("".to_string());
                                                }
                                                class=move || format!(
                                                    "w-full text-left px-4 py-2.5 text-xs transition-colors hover:bg-obsidian-accent/10 flex items-center justify-between {}",
                                                    if is_active() { "text-obsidian-accent bg-obsidian-accent/5 font-bold" } else { "text-obsidian-text/70" }
                                                )
                                            >
                                                <span class="truncate">{name}</span>
                                                <Show when=is_active>
                                                    <svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                                        <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                                                    </svg>
                                                </Show>
                                            </button>
                                        }
                                    }
                                />
                                {move || if filtered_categories.get().is_empty() {
                                    view! { <div class="px-4 py-3 text-xs text-obsidian-text/30 italic text-center">"No categories found"</div> }.into_any()
                                } else {
                                    ().into_any()
                                }}
                            </div>
                        </div>
                    </Show>
            </div>

            <div class="flex flex-col gap-2 flex-1 min-w-[140px]">
                <label class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-text/20 ml-2">"From"</label>
                <input
                    type="date"
                    class="bg-black/20 border border-white/5 rounded-2xl px-4 py-3 text-xs text-obsidian-heading focus:outline-none focus:ring-2 focus:ring-obsidian-accent/30 transition-all font-medium"
                    on:input=move |ev| set_date_filter.set(event_target_value(&ev))
                    on:keydown=move |ev| {
                        if ev.key() == "Enter" {
                            on_search.run(());
                        }
                    }
                    prop:value=date_filter
                />
            </div>

            <div class="flex flex-col gap-2 flex-1 min-w-[140px]">
                <label class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-text/20 ml-2">"Until"</label>
                <input
                    type="date"
                    class="bg-black/20 border border-white/5 rounded-2xl px-4 py-3 text-xs text-obsidian-heading focus:outline-none focus:ring-2 focus:ring-obsidian-accent/30 transition-all font-medium"
                    on:input=move |ev| set_end_date_filter.set(event_target_value(&ev))
                    on:keydown=move |ev| {
                        if ev.key() == "Enter" {
                            on_search.run(());
                        }
                    }
                    prop:value=end_date_filter
                />
            </div>

            <div class="flex flex-col gap-2 min-w-[200px]">
                <label class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-text/20 ml-2">"Results Per Page"</label>
                <div class="flex gap-2">
                    <input
                        type="number"
                        min="1"
                        class="flex-1 bg-black/20 border border-white/5 rounded-2xl px-4 py-3 text-xs text-obsidian-heading focus:outline-none focus:ring-2 focus:ring-obsidian-accent/30 transition-all font-bold tracking-tight text-center"
                        on:input=move |ev| {
                            if let Ok(n) = event_target_value(&ev).parse::<usize>() {
                                if n > 0 {
                                    set_page_size.set(n);
                                }
                            }
                        }
                        on:blur=move |_| {
                            // Validate and adjust based on view mode when user finishes typing
                            let current = page_size.get();
                            if view_mode.get() == "card" && current % 3 != 0 {
                                // Round to nearest multiple of 3
                                let adjusted = ((current + 1) / 3) * 3;
                                if adjusted > 0 {
                                    set_page_size.set(adjusted);
                                }
                            }
                        }
                        prop:value=move || page_size.get().to_string()
                    />
                    <div class="relative group/presets">
                        <button
                            class="p-3 bg-black/20 border border-white/5 rounded-2xl hover:border-obsidian-accent/30 transition-all"
                            title="Presets"
                        >
                            <svg class="w-4 h-4 text-obsidian-text/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                        </button>
                        <div class="hidden group-hover/presets:block absolute right-0 mt-2 bg-obsidian-card border border-white/10 rounded-2xl shadow-2xl p-2 z-[100] min-w-[140px]">
                            <div class="text-[9px] font-black text-obsidian-text/30 uppercase tracking-wider px-3 py-2">"Quick Select"</div>
                            {move || {
                                let presets = if view_mode.get() == "card" {
                                    vec![12, 24, 51, 99, 201]
                                } else {
                                    vec![10, 25, 50, 100, 200]
                                };
                                presets.into_iter().map(|size| {
                                    view! {
                                        <button
                                            on:click=move |_| set_page_size.set(size)
                                            class="w-full text-left px-3 py-2 text-xs text-obsidian-text/70 hover:bg-obsidian-accent/10 hover:text-obsidian-accent rounded-lg transition-colors font-semibold"
                                        >
                                            {format!("{} results", size)}
                                        </button>
                                    }
                                }).collect_view()
                            }}
                        </div>
                    </div>
                </div>
                <Show when=move || { view_mode.get() == "card" && page_size.get() % 3 != 0 }>
                    <p class="text-[9px] text-amber-400/80 ml-2 font-semibold">"⚠ Grid view requires multiples of 3"</p>
                </Show>
            </div>
            <div class="flex flex-col gap-2 min-w-[340px] flex-[3]">
                <div class="flex items-center justify-between ml-2">
                    <label class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-text/20">"Intelligent Exclusion Filter"</label>
                    <div class="flex items-center gap-2 cursor-pointer group/check" on:click=move |_| set_use_llm.update(|v| *v = !*v)>
                         <span class="text-[9px] font-black text-obsidian-text/20 group-hover/check:text-obsidian-accent transition-colors tracking-widest uppercase">"Neural Filter"</span>
                         <div class=move || format!("w-10 h-5 rounded-full transition-all relative border border-white/5 {}", if use_llm.get() { "bg-obsidian-accent/20" } else { "bg-black/20" })>
                            <div class=move || format!("absolute top-1 left-1 w-3 h-3 rounded-full transition-all shadow-sm {}", if use_llm.get() { "translate-x-5 bg-obsidian-accent text-glow" } else { "translate-x-0 bg-white/20" })></div>
                         </div>
                    </div>
                </div>
                <div class="relative group">
                    <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                        <svg class="h-4 w-4 text-obsidian-text/10 group-focus-within:text-red-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M18.364 5.636l-12.728 12.728m0-12.728l12.728 12.728" />
                        </svg>
                    </div>
                    <input
                        type="text"
                        placeholder=move || if use_llm.get() { "Topics to hide (AI-powered dynamic removal)..." } else { "Fixed keyword exclusion..." }
                        class="w-full bg-black/20 border border-white/5 rounded-2xl pl-11 pr-4 py-3 text-sm text-obsidian-heading focus:outline-none focus:ring-2 focus:ring-red-500/20 focus:border-red-500/30 transition-all font-medium placeholder:text-obsidian-text/10 shadow-inner"
                        on:input=move |ev| set_negative_query.set(event_target_value(&ev))
                        on:keydown=move |ev| {
                            if ev.key() == "Enter" {
                                on_search.run(());
                            }
                        }
                        prop:value=negative_query
                    />
                </div>
            </div>

            <div class="hidden lg:block w-[1px] h-10 bg-white/5 mx-2"></div>

            <div class="flex flex-wrap items-center gap-3 w-full lg:w-auto self-end justify-center lg:justify-end">
                <button
                    on:click=move |_| on_search.run(())
                    class="flex-1 lg:flex-none h-[46px] px-8 bg-gradient-to-br from-obsidian-accent to-obsidian-accent/80 text-white text-[11px] font-black uppercase tracking-[0.2em] rounded-2xl hover:brightness-110 transition-all shadow-[0_10px_20px_-10px_rgba(59,130,246,0.6)] active:scale-95 flex items-center justify-center gap-2.5"
                >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    "Search"
                </button>

                <button
                    on:click=move |_| on_reset.run(())
                    class="flex-1 lg:flex-none h-[46px] px-8 bg-white/5 text-obsidian-text text-[11px] font-black uppercase tracking-[0.2em] rounded-2xl hover:bg-white/10 transition-all border border-white/5 active:scale-95 flex items-center justify-center gap-2.5"
                >
                    <svg class="w-4 h-4 text-obsidian-text/20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    "Reset"
                </button>

                <button
                    on:click=move |_| on_fetch.run(())
                    disabled=move || fetch_pending.get()
                    class="flex-1 lg:flex-none h-[46px] px-8 bg-white/5 text-obsidian-text text-[11px] font-black uppercase tracking-[0.2em] rounded-2xl hover:bg-white/10 transition-all border border-white/5 active:scale-95 flex items-center justify-center gap-2.5 disabled:opacity-30 disabled:cursor-not-allowed group"
                >
                    <Show
                        when=move || fetch_pending.get()
                        fallback=move || view! {
                            <svg class="w-4 h-4 text-obsidian-accent group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                            </svg>
                            "Fetch New"
                        }
                    >
                        <div class="w-4 h-4 border-2 border-white/10 border-t-obsidian-accent rounded-full animate-spin"></div>
                        "Syncing..."
                    </Show>
                </button>

                <button
                    on:click=move |_| on_edit_config.run(())
                    class="h-[46px] px-6 bg-white/5 text-obsidian-text text-[11px] font-black uppercase tracking-[0.2em] rounded-2xl hover:bg-white/10 transition-all flex items-center justify-center gap-2.5 border border-white/5"
                    title="Edit Configuration"
                >
                    <svg class="w-4 h-4 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.726-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    "Config"
                </button>

                <button
                    on:click=move |_| on_about.run(())
                    class="h-[46px] px-5 bg-white/5 text-obsidian-text text-[11px] font-black uppercase tracking-[0.2em] rounded-2xl hover:bg-white/10 transition-all flex items-center justify-center gap-2.5 border border-white/5"
                    title="About"
                >
                    <svg class="w-4 h-4 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    "About"
                </button>

                <button
                    on:click=move |_| on_surprise.run(())
                    class="h-[46px] px-5 bg-obsidian-accent/10 text-obsidian-accent text-[11px] font-black uppercase tracking-[0.2em] rounded-2xl hover:bg-obsidian-accent/20 transition-all flex items-center justify-center gap-2.5 border border-obsidian-accent/20"
                    title="Unlock Secret Surprise"
                >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                    </svg>
                    "Surprise"
                </button>

                <button
                    on:click=move |_| on_unload_model.run(())
                    class="h-[46px] px-6 bg-red-500/5 text-red-400 text-[11px] font-black uppercase tracking-[0.2em] rounded-2xl hover:bg-red-500/10 transition-all flex items-center justify-center gap-2.5 border border-red-500/10"
                >
                    <svg class="w-4 h-4 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    "Flush"
                </button>
            </div>
        </div>
    }
}

#[component]

fn BackToTop() -> impl IntoView {

    let (show, set_show) =
        signal(false);

    #[cfg(feature = "ssr")]
    let _ = set_show;

    #[cfg(feature = "hydrate")]
    {

        use leptos::ev;
        use leptos::prelude::window_event_listener;

        // Debug hydration
        log::info!(
            "BackToTop hydrated"
        );

        window_event_listener(
            ev::scroll,
            move |_| {
                if let Some(window) =
                    web_sys::window()
                {

                    let y = window
                        .scroll_y()
                        .unwrap_or(0.0);

                    let show_now =
                        y > 300.0;

                    if show_now != show.get_untracked() {
                    set_show.set(show_now);
                }
                }
            },
        );
    }

    // Monitor signal changes
    Effect::new(move |_| {

        log::info!(
            "BackToTop visibility: {}",
            show.get()
        );
    });

    let scroll_to_top = move |_| {

        #[cfg(feature = "hydrate")]
        if let Some(window) =
            web_sys::window()
        {

            let options = web_sys::ScrollToOptions::new();

            options.set_top(0.0);

            options.set_behavior(web_sys::ScrollBehavior::Smooth);

            window.scroll_to_with_scroll_to_options(&options);
        }
    };

    view! {
        <div class="fixed bottom-12 right-12 z-[9999]" style="pointer-events: none;">
            <Show when=move || show.get()>
                <button
                    on:click=scroll_to_top
                    class="w-14 h-14 bg-gradient-to-br from-obsidian-accent to-obsidian-accent-light text-white rounded-2xl shadow-[0_20px_40px_-10px_rgba(59,130,246,0.5)] hover:bg-obsidian-accent-light hover:scale-110 active:scale-90 transition-all duration-500 group flex items-center justify-center border border-white/20 animate-scale-in"
                    style="pointer-events: auto;"
                    title="Back to Top"
                >
                    <div class="flex flex-col items-center">
                        <svg class="w-6 h-6 group-hover:-translate-y-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 15l7-7 7 7" />
                        </svg>
                    </div>
                </button>
            </Show>
        </div>
    }
}

#[component]

pub fn AboutModal(
    show: Signal<bool>,
    on_close: Callback<()>,
) -> impl IntoView {

    let version_resource =
        Resource::new(
            move || show.get(),
            |show_val| {

                async move {

                    if show_val {

                        get_version_info().await
                    } else {

                        Ok(VersionInfo::default())
                    }
                }
            },
        );

    view! {
        <Show when=move || show.get()>
            <div class="fixed inset-0 z-[100] flex items-center justify-center p-6 bg-black/40 backdrop-blur-md animate-fade-in" on:click=move |_| on_close.run(())>
                <div class="glass-dark border border-white/10 rounded-[2.5rem] w-full max-w-lg max-h-[85vh] flex flex-col shadow-[0_50px_100px_-20px_rgba(0,0,0,0.5)] overflow-hidden animate-scale-in" on:click=|ev| ev.stop_propagation()>
                    <div class="px-8 py-6 border-b border-white/5 flex justify-between items-center">
                        <div class="flex items-center gap-3">
                            <div class="p-2 bg-obsidian-accent/10 rounded-xl">
                                <svg class="w-5 h-5 text-obsidian-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </div>
                            <h2 class="text-xl font-black text-obsidian-heading tracking-tight">"Metadata"</h2>
                        </div>
                        <button on:click=move |_| on_close.run(()) class="w-10 h-10 rounded-full flex items-center justify-center text-obsidian-text/40 hover:bg-white/5 hover:text-white transition-all">"✕"</button>
                    </div>
                    <div class="p-8 space-y-8 overflow-y-auto flex-1 custom-scrollbar">
                        <div class="flex flex-col items-center text-center space-y-4">
                            <div class="relative group/logo">
                                <div class="absolute -inset-4 bg-gradient-to-r from-obsidian-accent/20 to-purple-500/20 rounded-full blur-2xl opacity-50"></div>
                                <img src="/logo.svg" alt="Logo" class="relative w-24 h-24 rounded-3xl shadow-2xl shadow-[0_0_30px_rgba(59,130,246,0.5)] bg-black/40 p-1" />
                            </div>
                            <div>
                                <h3 class="text-3xl font-black text-white italic tracking-tighter">"arXiv" <span class="text-obsidian-accent gradient-text text-glow">"Daily"</span></h3>
                                <p class="text-[10px] text-obsidian-text/20 font-black uppercase tracking-[0.3em] mt-2">"Neural Discovery Platform"</p>
                            </div>
                        </div>

                        <div class="space-y-4">
                            <Transition fallback=move || view! { <div class="p-8 text-center animate-pulse text-obsidian-text/10 text-xs font-black uppercase tracking-widest">"Accessing Build Records..."</div> }>
                                {move || version_resource.get().map(|res| {
                                    match res {
                                        Ok(info) => view! {
                                            <div class="space-y-6">
                                                <div class="glass bg-white/2 rounded-[1.5rem] p-6 space-y-5">
                                                    <h4 class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-accent/50 ml-1">"Artifact Specifications"</h4>
                                                    <div class="space-y-4">
                                                        <div class="flex justify-between items-center">
                                                            <span class="text-xs font-bold text-obsidian-text/30 uppercase tracking-widest">"Revision"</span>
                                                            <span class="text-xs font-black text-obsidian-accent bg-obsidian-accent/5 px-3 py-1 rounded-full border border-obsidian-accent/10">{info.build_semver}</span>
                                                        </div>
                                                        <div class="flex justify-between items-center">
                                                            <span class="text-xs font-bold text-obsidian-text/30 uppercase tracking-widest">"Compiler"</span>
                                                            <span class="text-xs font-bold text-obsidian-heading italic">{info.authors}</span>
                                                        </div>
                                                        <div class="flex flex-col gap-2">
                                                            <span class="text-xs font-bold text-obsidian-text/30 uppercase tracking-widest">"Source Path"</span>
                                                            <span class="text-[10px] font-mono text-obsidian-heading/60 bg-black/20 p-2 rounded-xl border border-white/5 truncate">{info.repository}</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div class="glass bg-white/2 rounded-[1.5rem] p-6 space-y-5">
                                                    <h4 class="text-[10px] font-black uppercase tracking-[0.2em] text-amber-500/50 ml-1">"Deployment Metadata"</h4>
                                                    <div class="space-y-4">
                                                        <div class="flex justify-between items-center">
                                                            <span class="text-xs font-bold text-obsidian-text/30 uppercase tracking-widest">"Timestamp"</span>
                                                            <span class="text-[11px] font-black text-obsidian-heading/80">{info.build_timestamp}</span>
                                                        </div>
                                                        <div class="flex justify-between items-center">
                                                            <span class="text-xs font-bold text-obsidian-text/30 uppercase tracking-widest">"Level"</span>
                                                            <span class="text-[11px] font-black text-amber-500/80">"O"{info.optimization}</span>
                                                        </div>
                                                        <div class="flex justify-between items-center">
                                                            <span class="text-xs font-bold text-obsidian-text/30 uppercase tracking-widest">"Symbols"</span>
                                                            <span class="text-[11px] font-black text-obsidian-heading/80 uppercase">{info.debug_symbols}</span>
                                                        </div>
                                                        <div class="flex flex-col gap-1">
                                                            <span class="text-obsidian-text/40 text-xs">"Target Triple"</span>
                                                            <span class="text-obsidian-heading font-mono text-[10px] truncate">{info.target_triple}</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div class="p-6 space-y-4">
                                                    <h4 class="text-[10px] font-black uppercase tracking-[0.2em] text-purple-400">"Git Telemetry"</h4>
                                                    <div class="space-y-3 text-sm">
                                                        <div class="flex justify-between">
                                                            <span class="text-obsidian-text/40">"Branch"</span>
                                                            <span class="text-obsidian-heading font-mono text-xs">{info.git_branch}</span>
                                                        </div>
                                                        <div class="flex justify-between">
                                                            <span class="text-obsidian-text/40">"Commit SHA"</span>
                                                            <span class="text-obsidian-heading font-mono text-[10px]">{info.git_sha}</span>
                                                        </div>
                                                        <div class="flex flex-col gap-1">
                                                            <span class="text-obsidian-text/40 text-xs">"Author"</span>
                                                            <span class="text-obsidian-heading text-[10px] italic">{info.git_author}</span>
                                                        </div>
                                                        <div class="flex flex-col gap-1">
                                                            <span class="text-obsidian-text/40 text-xs">"Commit Message"</span>
                                                            <span class="text-white/60 text-[10px] bg-white/5 p-2 rounded-lg font-mono leading-relaxed line-clamp-3 italic">
                                                                {info.git_commit_message}
                                                            </span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div class="p-6 space-y-4">
                                                    <h4 class="text-[10px] font-black uppercase tracking-[0.2em] text-blue-400">"Compiler Info"</h4>
                                                    <div class="space-y-3 text-sm">
                                                        <div class="flex justify-between">
                                                            <span class="text-obsidian-text/40">"Rustc"</span>
                                                            <span class="text-obsidian-heading font-mono text-xs">{info.rustc_semver}</span>
                                                        </div>
                                                        <div class="flex justify-between">
                                                            <span class="text-obsidian-text/40">"LLVM"</span>
                                                            <span class="text-obsidian-heading font-mono text-xs">{info.llvm_version}</span>
                                                        </div>
                                                        <div class="flex flex-col gap-1">
                                                            <span class="text-obsidian-text/40 text-xs">"Host Triple"</span>
                                                            <span class="text-obsidian-heading font-mono text-[10px] truncate">{info.host_triple}</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div class="p-6 space-y-4">
                                                    <h4 class="text-[10px] font-black uppercase tracking-[0.2em] text-emerald-400">"Build Host Specs"</h4>
                                                    <div class="space-y-3 text-sm">
                                                        <div class="flex justify-between">
                                                            <span class="text-obsidian-text/40">"OS Target"</span>
                                                            <span class="text-emerald-400/80 italic text-xs font-bold">{info.os_name} " ("{info.os_version}")"</span>
                                                        </div>
                                                        <div class="flex flex-col gap-1">
                                                            <span class="text-obsidian-text/40 text-xs">"Kernel Version"</span>
                                                            <span class="text-obsidian-heading font-mono text-[10px] truncate">{info.kernel_version}</span>
                                                        </div>
                                                        <div class="flex flex-col gap-1">
                                                            <span class="text-obsidian-text/40 text-xs">"CPU Processor"</span>
                                                            <span class="text-obsidian-heading font-mono text-[10px] truncate">{info.cpu_brand}</span>
                                                        </div>
                                                        <div class="flex justify-between">
                                                            <span class="text-obsidian-text/40">"Cores"</span>
                                                            <span class="text-obsidian-heading font-mono">{info.cpu_cores}</span>
                                                        </div>
                                                        <div class="flex justify-between">
                                                            <span class="text-obsidian-text/40">"Total RAM"</span>
                                                            <span class="text-emerald-400/80 font-mono font-bold">{info.total_memory}</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        }.into_any(),
                                        Err(e) => view! { <div class="p-4 text-red-400 text-xs font-mono bg-red-500/5 rounded-lg border border-red-500/10">"Error: "{e.to_string()}</div> }.into_any(),
                                    }
                                })}
                            </Transition>
                        </div>

                        <p class="text-[10px] text-center text-obsidian-text/20 uppercase font-black tracking-[0.3em] pt-4 leading-relaxed">
                            "Designed for researchers by" <br/>
                            <span class="text-obsidian-text/40">"APICH ORGANIZATION"</span>
                        </p>
                    </div>
                    <div class="p-4 bg-white/5 flex justify-center">
                         <button
                            on:click=move |_| on_close.run(())
                            class="w-full py-3 text-xs font-black uppercase tracking-widest text-obsidian-text/60 hover:text-white transition-colors"
                        >
                            "Close System Info"
                        </button>
                    </div>
                </div>
            </div>
        </Show>
    }
}

#[component]

fn PaperListItem(
    paper: Paper
) -> impl IntoView {

    let (expanded, set_expanded) =
        signal(false);

    let authors = paper.authors_list();

    let display_authors =
        authors.join(", ");

    let title = paper.title.clone();

    let url = paper.url.clone();

    let summary = paper
        .summary
        .clone();

    let pdf_link = paper
        .pdf_link
        .clone();

    let published_str = paper
        .published_date()
        .format("%b %d, %Y")
        .to_string();

    let category_name =
        paper.primary_category_name();

    view! {
        <div class="glass glow-hover rounded-2xl p-6 flex items-start gap-6 hover:scale-[1.01] active:scale-[0.99] transition-all duration-300 group relative overflow-hidden">
            <div class="absolute -left-12 top-0 bottom-0 w-24 bg-obsidian-accent/5 blur-[60px] group-hover:bg-obsidian-accent/10 transition-colors"></div>

            <div class="flex-1 space-y-4 relative z-10">
                <div class="flex items-start justify-between gap-4">
                    <div class="flex-1">
                        <div class="flex items-center gap-3 mb-2">
                            <span class="inline-flex items-center px-2.5 py-1 rounded-full text-[9px] font-black bg-obsidian-accent/5 text-obsidian-accent border border-obsidian-accent/10 uppercase tracking-[0.1em]">
                                {category_name}
                            </span>
                            <span class="text-[9px] uppercase tracking-[0.15em] text-obsidian-text/30 font-bold">
                                {published_str}
                            </span>
                        </div>
                        <h3 class="text-lg font-bold text-obsidian-heading leading-[1.3] group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-obsidian-accent group-hover:to-obsidian-accent-light transition-all duration-500 mb-2">
                            <a
                                href=url.clone()
                                target="_blank"
                                class="hover:underline decoration-obsidian-accent/20 decoration-2 underline-offset-4 transition-all"
                                inner_html=move || render_math(&title)
                            ></a>
                        </h3>
                        <p class="text-[10px] text-obsidian-text/30 font-semibold mb-3 line-clamp-1 italic tracking-wide">
                            {display_authors}
                        </p>
                        <div class="space-y-2">
                            <p class=move || if expanded.get() {
                                "text-sm text-obsidian-text/60 leading-relaxed font-medium"
                            } else {
                                "text-sm text-obsidian-text/60 leading-relaxed line-clamp-2 font-medium"
                            }>
                                {summary.clone()}
                            </p>
                            <Show when=move || { summary.len() > 200 }>
                                <button
                                    on:click=move |_| set_expanded.update(|e| *e = !*e)
                                    class="text-xs text-obsidian-accent hover:text-obsidian-accent-light font-semibold transition-colors flex items-center gap-1"
                                >
                                    {move || if expanded.get() { "Show less" } else { "Show more" }}
                                    <svg class=move || format!("w-3 h-3 transition-transform {}", if expanded.get() { "rotate-180" } else { "" }) fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M19 9l-7 7-7-7" />
                                    </svg>
                                </button>
                            </Show>
                        </div>
                    </div>

                    <div class="flex gap-2 shrink-0">
                        <a
                            href=url
                            target="_blank"
                            class="p-2.5 bg-obsidian-accent/10 hover:bg-obsidian-accent/20 text-obsidian-accent rounded-xl transition-all hover:scale-110 active:scale-95 border border-obsidian-accent/20"
                            title="View on arXiv"
                        >
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                        </a>
                        <a
                            href=pdf_link
                            target="_blank"
                            class="p-2.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-xl transition-all hover:scale-110 active:scale-95 border border-red-500/20"
                            title="Download PDF"
                        >
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    }
}

#[cfg(feature = "hydrate")]
#[wasm_bindgen::prelude::wasm_bindgen]

pub fn hydrate() {

    use console_error_panic_hook;
    use console_log;

    _ = console_log::init_with_level(
        log::Level::Debug,
    );

    console_error_panic_hook::set_once(
    );

    leptos::mount::hydrate_body(App);
}
