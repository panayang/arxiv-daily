//! This is the library for the arxiv-daily project.

#![allow(
    clippy::empty_line_after_outer_attr
)]
#![recursion_limit = "4096"]

#[cfg(feature = "ssr")]
use std::sync::Arc;

use bitcode::Decode;
use bitcode::Encode;
use leptos::prelude::*;
use leptos::server_fn::codec::Bitcode;
use leptos_meta::*;
use leptos_router::components::*;
use leptos_router::*;
use serde::Deserialize;
use serde::Serialize;
use shared::Category;
use shared::Paper;
#[cfg(feature = "ssr")]
use tokio::sync::Mutex;
#[allow(unused_imports)]
#[cfg(feature = "ssr")]
use tokio::sync::OnceCell;

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

    let mut papers: Vec<Paper> = rows
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

        if use_llm {

            papers = filter_with_ai(
                papers,
                negative_query,
            )
            .await;
        } else {

            papers =
                filter_with_keywords(
                    papers,
                    negative_query,
                );
        }
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

#[server(GetConfig, "/api", input = Bitcode, output = Bitcode)]

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


#[server(SaveConfig, "/api", input = Bitcode, output = Bitcode)]

pub async fn save_config(
    content: String
) -> Result<(), ServerFnError> {

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

#[server(GetVersionInfo, "/api", input = Bitcode, output = Bitcode)]

pub async fn get_version_info()
-> Result<VersionInfo, ServerFnError> {

    Ok(VersionInfo {
        build_semver: env!("CARGO_PKG_VERSION").to_string(),
        authors: env!("CARGO_PKG_AUTHORS").to_string(),
        repository: env!("VERGEN_GIT_DESCRIBE").to_string(),
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
            s.replace("-", "") + "0600";

        let e_fmt = end_date
            .replace("-", "")
            + "0600";

        // ArXiv API requires at least one search term (like cat:xxx) to be used with a date range.
        // We use the target category to satisfy this requirement.
        format!(
            "https://export.arxiv.org/api/query?search_query=cat:{}+AND+submittedDate:[{}+TO+{}]&start={}&max_results={}&sortBy=submittedDate&sortOrder=descending",
            target_category, s_fmt, e_fmt, config.arxiv.start, config.arxiv.max_results
        )
    } else {

        format!(
            "http://export.arxiv.org/api/query?search_query=cat:{}&start={}&max_results={}&sortBy=submittedDate&sortOrder=descending",
            target_category, config.arxiv.start, config.arxiv.max_results
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

#[cfg(feature = "ssr")]

static MODEL: Mutex<
    Option<Arc<Mutex<ai::Gemma3>>>,
> = Mutex::const_new(None);

#[server(UnloadModel, "/api", input = Bitcode, output = Bitcode)]

pub async fn unload_model()
-> Result<(), ServerFnError> {

    log::info!(
        "Unloading AI model from \
         RAM..."
    );

    let mut model = MODEL.lock().await;

    *model = None;

    Ok(())
}

#[cfg(feature = "ssr")]

async fn filter_with_ai(
    papers: Vec<Paper>,
    negative_query: String,
) -> Vec<Paper> {

    if negative_query.is_empty()
        || papers.is_empty()
    {

        return papers;
    }

    log::info!(
        "AI filtering {} papers with \
         negative filter: '{}'",
        papers.len(),
        negative_query
    );

    let mut model_lock =
        MODEL.lock().await;

    let model_mu = if let Some(m) =
        &*model_lock
    {

        m.clone()
    } else {

        let (
            model_path,
            tokenizer_path,
        ) = if std::path::Path::new(
            "assets/llm.bin",
        )
        .exists()
        {

            (
                "assets/llm.bin"
                    .to_string(),
                "assets/tokenizer.bin"
                    .to_string(),
            )
        } else if std::path::Path::new(
            "../assets/llm.bin",
        )
        .exists()
        {

            (
                "../assets/llm.bin"
                    .to_string(),
                "../assets/tokenizer.\
                 bin"
                .to_string(),
            )
        } else {

            (
                "assets/llm.bin"
                    .to_string(),
                "assets/tokenizer.bin"
                    .to_string(),
            )
        };

        log::info!(
            "Loading custom Gemma 3 \
             model from {}...",
            model_path
        );

        let mut gemma =
            ai::Gemma3::new(
                model_path,
                tokenizer_path,
            )
            .expect(
                "Failed to load Gemma \
                 3 model",
            );

        // Run self-test
        // gemma.self_test().expect("AI Self-test failed");
        gemma.set_initialized(true);

        let m =
            Arc::new(Mutex::new(gemma));

        *model_lock = Some(m.clone());

        m
    };

    drop(model_lock);

    let mut model = model_mu
        .lock()
        .await;

    let mut filtered_papers =
        Vec::new();

    for paper in papers {

        let summary_snippet = if paper
            .summary
            .len()
            > 300
        {

            format!(
                "{}...",
                &paper.summary
                    [0 .. 300]
            )
        } else {

            paper
                .summary
                .clone()
        };

        let prompt = format!(
            "<start_of_turn>user\\
             nInstructions: Respond \
             ONLY with YES or \
             NO.\nNegative Filter: \
             Skip papers related to \
             \"{}\".\nPaper Title: \
             {}\nSummary Snippet: \
             {}\nQuestion: Should I \
             skip this paper?\nAnswer: \
             <end_of_turn>\\
             n<start_of_turn>model\n",
            negative_query,
            paper.title,
            summary_snippet
        );

        let response = match model
            .complete(&prompt, 3)
        {
            | Ok(r) => r,
            | Err(e) => {

                log::error!(
                    "AI completion \
                     error: {}",
                    e
                );

                filtered_papers
                    .push(paper);

                continue;
            },
        };

        let response = response
            .trim()
            .to_uppercase();

        let is_excluded =
            response.contains("YES");

        if !is_excluded {

            filtered_papers.push(paper);
        } else {

            log::info!(
                "AI Filtered out: {}",
                paper.title
            );
        }
    }

    filtered_papers
}

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

#[component]

pub fn App() -> impl IntoView {

    provide_meta_context();

    view! {
        <Stylesheet id="leptos" href="/pkg/web_app.css"/>
        <Title text="arXiv Dashboard"/>

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

    let (use_llm, set_use_llm) =
        signal(true);

    let (page, set_page) =
        signal(1usize);

    let (show_config, set_show_config) =
        signal(false);

    let (show_about, set_show_about) =
        signal(false);

    #[derive(
        Clone, Default, PartialEq,
    )]

    struct SearchParams {
        query: String,
        category: String,
        date: String,
        end_date: String,
        page: usize,
        negative_query: String,
        use_llm: bool,
    }

    let (
        trigger_search,
        set_trigger_search,
    ) = signal(SearchParams {
        query: "".to_string(),
        category: "all".to_string(),
        date: "".to_string(),
        end_date: "".to_string(),
        page: 1,
        negative_query: "".to_string(),
        use_llm: true,
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
                    51,
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

    // Auto-search on category/date change
    Effect::new(move |_| {

        let category =
            selected_category.get();

        let date = date_filter.get();

        let end_date =
            end_date_filter.get();

        set_trigger_search.update(
            |p| {

                p.category = category;

                p.date = date;

                p.end_date = end_date;

                p.page = 1; // Reset page on filter change
            },
        );

        set_page.set(1);
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
                negative_query: ""
                    .to_string(),
                use_llm: true,
            },
        );
    };

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
            <header class="flex flex-col md:flex-row md:items-end justify-between gap-6 border-b border-white/10 pb-10">
                <div class="space-y-2">
                    <div class="flex items-center gap-3">
                        <img src="/logo.svg" alt="arXiv Daily Logo" class="h-10 w-10 rounded-lg shadow-lg shadow-obsidian-accent/20" />
                        <h1 class="text-4xl font-extrabold text-obsidian-heading tracking-tighter">"arXiv" <span class="text-obsidian-accent">"Daily"</span></h1>
                    </div>
                    <p class="text-obsidian-text/50 font-medium ml-1">"Personalized research discovery platform"</p>
                </div>
                <div class="relative w-full md:w-96 group">
                    <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <svg class="h-5 w-5 text-obsidian-text/30 group-focus-within:text-obsidian-accent transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                    </div>
                    <input
                        type="text"
                        placeholder="Search research papers..."
                        class="w-full bg-obsidian-sidebar border border-white/10 rounded-xl pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-obsidian-accent/40 focus:border-obsidian-accent/40 transition-all text-obsidian-heading placeholder:text-obsidian-text/25 shadow-inner"
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
            </header>

            {move || {
                let pending = fetch_action.pending();
                let value = fetch_action.value();

                view! {
                    <Show when=move || show_fetch_status.get()>
                        <div class="fixed top-8 left-1/2 -translate-x-1/2 z-[100] animate-in fade-in slide-in-from-top-4 duration-300">
                            <div class="bg-obsidian-sidebar border border-white/10 rounded-2xl px-6 py-3 shadow-2xl flex items-center gap-4">
                                <Show
                                    when=move || pending.get()
                                    fallback=move || {
                                        value.with(|v| {
                                            match v {
                                                Some(Ok(count)) => {
                                                    let count = *count;
                                                    view! {
                                                        <div class="flex items-center gap-3">
                                                            <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                                                            <p class="text-sm font-bold text-white">"Successfully fetched "{count}" new papers"</p>
                                                        </div>
                                                    }.into_any()
                                                }
                                                Some(Err(e)) => view! {
                                                    <div class="flex items-center gap-3">
                                                        <div class="w-2 h-2 bg-red-500 rounded-full"></div>
                                                        <p class="text-sm font-bold text-red-200">"Fetch failed: "{e.to_string()}</p>
                                                    </div>
                                                }.into_any(),
                                                _ => ().into_any()
                                            }
                                        })
                                    }
                                >
                                    <div class="flex items-center gap-3">
                                        <div class="w-4 h-4 border-2 border-obsidian-accent/20 border-t-obsidian-accent rounded-full animate-spin"></div>
                                        <p class="text-sm font-bold text-obsidian-text">"Synchronizing with arXiv..."</p>
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
                on_unload_model=Callback::new(move |_| { unload_action.dispatch(()); })
                end_date_filter=end_date_filter.into()
                set_end_date_filter
                negative_query=negative_query.into()
                set_negative_query
                use_llm=use_llm.into()
                set_use_llm
            />

            <ConfigModal show=show_config.into() on_close=Callback::new(move |_| set_show_config.set(false))/>
            <AboutModal show=show_about.into() on_close=Callback::new(move |_| set_show_about.set(false))/>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 gap-8">
                <Transition fallback=move || view! {
                    <div class="col-span-full py-32 flex flex-col items-center justify-center space-y-4">
                        <div class="w-12 h-12 border-4 border-obsidian-accent/20 border-t-obsidian-accent rounded-full animate-spin"></div>
                        <p class="text-obsidian-text/40 font-medium animate-pulse">"Indexing archive databases..."</p>
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
                                    view! {
                                        <div class="col-span-full space-y-12">
                                            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 gap-8">
                                                <For
                                                    each=move || current_papers.clone()
                                                    key=|paper| paper.id.clone()
                                                    children=|paper| {
                                                        view! { <PaperCard paper=paper/> }
                                                    }
                                                />
                                            </div>
                                            <Pagination
                                                current_page=page.into()
                                                total_count=total_count
                                                page_size=51
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
        <div class="bg-obsidian-sidebar border border-white/5 rounded-2xl p-7 flex flex-col h-full hover:border-obsidian-accent/40 transition-all duration-300 group hover:shadow-2xl hover:shadow-black/40 relative overflow-hidden">
            <div class="absolute top-0 right-0 p-4 opacity-0 group-hover:opacity-100 transition-opacity">
                <div class="w-2 h-2 rounded-full bg-obsidian-accent animate-pulse"></div>
            </div>

            <div class="flex items-center justify-between gap-3 mb-5">
                <span class="inline-flex items-center px-3 py-1 rounded-md text-[10px] font-black bg-obsidian-accent/10 text-obsidian-accent border border-obsidian-accent/10 uppercase tracking-widest">
                    {category_name}
                </span>
                <span class="text-[10px] uppercase tracking-widest text-obsidian-text/30 font-bold">
                    {published_str}
                </span>
            </div>

            <h3 class="text-xl font-bold text-obsidian-heading leading-tight group-hover:text-obsidian-accent transition-colors mb-2">
                <a href=url.clone() target="_blank" class="hover:underline decoration-obsidian-accent/30 underline-offset-4">
                    {title}
                </a>
            </h3>

            <p class="text-xs text-obsidian-text/40 font-medium mb-6 line-clamp-1 italic group-hover:text-obsidian-text/60 transition-colors">
                {display_authors}
            </p>

            <div class="relative flex-grow">
                <div class=move || format!("text-sm leading-relaxed text-obsidian-text/70 transition-all duration-500 {}", if expanded.get() { "" } else { "line-clamp-4 overflow-hidden" })>
                    {summary.clone()}
                </div>
                {move || if !expanded.get() && summary.len() > 180 {
                    view! {
                        <button
                            on:click=move |_| set_expanded.set(true)
                            class="text-xs font-bold text-obsidian-accent hover:text-obsidian-accent/80 mt-3 flex items-center gap-1 group/btn transition-colors focus:outline-none"
                        >
                            "EXPLORE SUMMARY"
                            <svg class="w-3 h-3 group-hover/btn:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                            </svg>
                        </button>
                    }.into_any()
                } else {
                    ().into_any()
                }}
            </div>

            <div class="mt-8 flex items-center gap-4">
                {if let Some(link) = pdf_link {
                    view! {
                        <a
                            href=link
                            target="_blank"
                            class="flex-1 inline-flex items-center justify-center px-5 py-3 text-xs font-black text-white bg-obsidian-accent hover:bg-obsidian-accent/80 rounded-xl transition-all shadow-lg shadow-obsidian-accent/10 uppercase tracking-widest active:scale-95"
                        >
                            "View PDF"
                            <svg class="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                        </a>
                    }.into_any()
                } else {
                    view! {
                         <div class="flex-1 py-3 text-center text-[10px] text-obsidian-text/20 uppercase tracking-widest font-black border border-white/5 rounded-xl">"PDF Unavailable"</div>
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
) -> impl IntoView {

    let config_resource = Resource::new(
        move || show.get(),
        |_| {

            async move {

                get_config().await
            }
        },
    );

    let save_action = Action::new(
        |content: &String| {

            let content =
                content.clone();

            async move {

                save_config(content)
                    .await
            }
        },
    );

    let (content, set_content) =
        signal("".to_string());

    Effect::new(move |_| {
        if let Some(Ok(c)) =
            config_resource.get()
        {

            set_content.set(c);
        }
    });

    view! {
        <Show when=move || show.get()>
            <div class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
                <div class="bg-obsidian-sidebar border border-white/10 rounded-2xl w-full max-w-2xl shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
                    <div class="p-6 border-b border-white/5 flex justify-between items-center">
                        <div class="flex items-center gap-2">
                            <svg class="w-5 h-5 text-obsidian-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            </svg>
                            <h2 class="text-xl font-bold text-obsidian-heading">"Configuration Manager"</h2>
                        </div>
                        <button on:click=move |_| on_close.run(()) class="text-obsidian-text/40 hover:text-white transition-colors">"✕"</button>
                    </div>
                    <div class="p-6 space-y-4">
                        <label class="text-xs font-bold text-obsidian-text/40 uppercase tracking-widest">"Raw Config (TOML)"</label>
                        <textarea
                            class="w-full h-96 bg-obsidian-bg border border-white/10 rounded-xl p-4 font-mono text-sm text-obsidian-text focus:outline-none focus:ring-2 focus:ring-obsidian-accent/40"
                            on:input=move |ev| set_content.set(event_target_value(&ev))
                            prop:value=content
                        />
                        <div class="bg-blue-500/10 border border-blue-500/20 p-4 rounded-xl flex gap-3 italic">
                            <svg class="w-5 h-5 text-blue-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <p class="text-xs text-blue-200/70">"Warning: Changes to config.toml may require a server restart to take effect for all modules."</p>
                        </div>
                    </div>
                    <div class="p-6 border-t border-white/5 flex justify-end gap-3">
                        <button
                            on:click=move |_| on_close.run(())
                            class="px-5 py-2 text-sm font-bold text-obsidian-text/60 hover:text-white transition-colors"
                        >
                            "Cancel"
                        </button>
                        <button
                            on:click=move |_| {
                                save_action.dispatch(content.get());
                                on_close.run(());
                            }
                            class="px-6 py-2 bg-obsidian-accent text-white text-sm font-bold rounded-lg hover:bg-obsidian-accent/80 transition-all active:scale-95"
                        >
                            "Save Changes"
                        </button>
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
            <div class="flex items-center gap-2">
                <button
                    on:click=move |_| if can_go_prev() { on_page_change.run(current_page.get() - 1) }
                    disabled=move || !can_go_prev()
                    class="p-2 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 disabled:opacity-30 disabled:hover:bg-white/5 transition-all text-obsidian-text"
                >
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                    </svg>
                </button>

                <div class="flex items-center gap-2 font-mono text-sm px-4">
                    <span class="text-obsidian-text/30">"Page"</span>
                    <input
                        type="number"
                        min="1"
                        max=move || total_pages.get()
                        prop:value=move || current_page.get()
                        class="w-16 bg-white/5 border border-white/10 rounded-lg py-1 text-center font-bold text-obsidian-heading focus:outline-none focus:border-obsidian-accent/50 focus:ring-1 focus:ring-obsidian-accent/50 transition-all"
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
                    <span class="text-obsidian-text/30">"of"</span>
                    <span class="text-obsidian-text/50">{move || total_pages.get()}</span>
                </div>

                <button
                    on:click=move |_| if can_go_next() { on_page_change.run(current_page.get() + 1) }
                    disabled=move || !can_go_next()
                    class="p-2 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 disabled:opacity-30 disabled:hover:bg-white/5 transition-all text-obsidian-text"
                >
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
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
    on_unload_model: Callback<()>,
    use_llm: Signal<bool>,
    set_use_llm: WriteSignal<bool>,
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
        <div class="flex flex-wrap items-center gap-4 bg-obsidian-sidebar/50 p-4 rounded-2xl border border-white/5 shadow-inner">
            <div class="flex flex-col gap-1.5 min-w-[240px] flex-1 relative">
                <label class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-text/30 ml-1">"Category"</label>

                <div class="relative group">
                    <button
                        on:click=move |_| set_is_open.update(|v| *v = !*v)
                        class="w-full bg-obsidian-bg border border-white/10 rounded-xl px-4 py-2.5 text-sm text-obsidian-heading flex items-center justify-between hover:border-obsidian-accent/40 transition-all focus:outline-none focus:ring-2 focus:ring-obsidian-accent/40"
                    >
                        <span class="truncate">{current_category_name}</span>
                        <svg class=move || format!("w-4 h-4 text-obsidian-text/40 transition-transform duration-200 {}", if is_open.get() { "rotate-180" } else { "" }) fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                        </svg>
                    </button>

                    <Show when=move || is_open.get()>
                        <div class="absolute z-[60] mt-2 w-full bg-obsidian-sidebar border border-white/10 rounded-xl shadow-2xl overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200">
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
            </div>

            <div class="flex flex-col gap-1.5 flex-1 min-w-[150px]">
                <label class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-text/30 ml-1">"Start Date"</label>
                <input
                    type="date"
                    class="bg-obsidian-bg border border-white/10 rounded-xl px-4 py-2 text-sm text-obsidian-heading focus:outline-none focus:ring-2 focus:ring-obsidian-accent/40"
                    on:input=move |ev| set_date_filter.set(event_target_value(&ev))
                    on:keydown=move |ev| {
                        if ev.key() == "Enter" {
                            on_search.run(());
                        }
                    }
                    prop:value=date_filter
                />
            </div>

            <div class="flex flex-col gap-1.5 flex-1 min-w-[150px]">
                <label class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-text/30 ml-1">"End Date"</label>
                <input
                    type="date"
                    class="bg-obsidian-bg border border-white/10 rounded-xl px-4 py-2 text-sm text-obsidian-heading focus:outline-none focus:ring-2 focus:ring-obsidian-accent/40"
                    on:input=move |ev| set_end_date_filter.set(event_target_value(&ev))
                    on:keydown=move |ev| {
                        if ev.key() == "Enter" {
                            on_search.run(());
                        }
                    }
                    prop:value=end_date_filter
                />
            </div>

            <div class="flex flex-col gap-1.5 min-w-[300px] flex-[2]">
                <div class="flex items-center justify-between ml-1">
                    <label class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-text/30">"Negative Filter (What to hide)"</label>
                    <div class="flex items-center gap-2 cursor-pointer group/check" on:click=move |_| set_use_llm.update(|v| *v = !*v)>
                         <span class="text-[10px] font-bold text-obsidian-text/40 group-hover/check:text-obsidian-accent transition-colors">"USE LLM"</span>
                         <div class=move || format!("w-8 h-4 rounded-full transition-all relative {}", if use_llm.get() { "bg-obsidian-accent" } else { "bg-white/10" })>
                            <div class=move || format!("absolute top-1 left-1 w-2 h-2 bg-white rounded-full transition-all {}", if use_llm.get() { "translate-x-4" } else { "translate-x-0" })></div>
                         </div>
                    </div>
                </div>
                <div class="relative group">
                    <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <svg class="h-4 w-4 text-obsidian-text/20 group-focus-within:text-red-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 5.636l-12.728 12.728m0-12.728l12.728 12.728" />
                        </svg>
                    </div>
                    <input
                        type="text"
                        placeholder=move || if use_llm.get() { "AI will skip papers matching these topics..." } else { "Words to match (comma separated)..." }
                        class="w-full bg-obsidian-bg border border-white/10 rounded-xl pl-10 pr-4 py-2.5 text-sm text-obsidian-heading focus:outline-none focus:ring-2 focus:ring-red-500/20 focus:border-red-500/40 transition-all placeholder:text-obsidian-text/20 shadow-inner"
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

            <div class="flex items-center gap-3 w-full sm:w-auto self-end">
                <button
                    on:click=move |_| on_search.run(())
                    class="flex-1 sm:flex-none h-11 px-6 bg-obsidian-accent text-white text-xs font-black uppercase tracking-widest rounded-xl hover:bg-obsidian-accent/80 transition-all shadow-lg shadow-obsidian-accent/20 active:scale-95 flex items-center justify-center gap-2"
                >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    "Search"
                </button>

                <button
                    on:click=move |_| on_reset.run(())
                    class="flex-1 sm:flex-none h-11 px-6 bg-white/5 text-obsidian-text text-xs font-black uppercase tracking-widest rounded-xl hover:bg-white/10 transition-all border border-white/5 active:scale-95 flex items-center justify-center gap-2"
                >
                    <svg class="w-4 h-4 text-obsidian-text/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    "Reset"
                </button>

                <button
                    on:click=move |_| on_fetch.run(())
                    disabled=move || fetch_pending.get()
                    class="flex-1 sm:flex-none h-11 px-6 bg-white/5 text-obsidian-text text-xs font-black uppercase tracking-widest rounded-xl hover:bg-white/10 transition-all border border-white/5 active:scale-95 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <Show
                        when=move || fetch_pending.get()
                        fallback=move || view! {
                            <svg class="w-4 h-4 text-obsidian-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                            </svg>
                            "Fetch New"
                        }
                    >
                        <div class="w-4 h-4 border-2 border-white/10 border-t-obsidian-accent rounded-full animate-spin"></div>
                        "Fetching..."
                    </Show>
                </button>

                <button
                    on:click=move |_| on_edit_config.run(())
                    class="h-11 px-5 bg-white/5 text-obsidian-text text-xs font-black uppercase tracking-widest rounded-xl hover:bg-white/10 transition-all flex items-center justify-center gap-2"
                    title="Edit Configuration"
                >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    "Config"
                </button>

                <button
                    on:click=move |_| on_about.run(())
                    class="h-11 px-5 bg-white/5 text-obsidian-text text-xs font-black uppercase tracking-widest rounded-xl hover:bg-white/10 transition-all flex items-center justify-center gap-2"
                    title="About"
                >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    "About"
                </button>

                <button
                    on:click=move |_| on_unload_model.run(())
                    class="h-11 px-5 bg-white/5 text-red-400 text-xs font-black uppercase tracking-widest rounded-xl hover:bg-red-500/10 transition-all flex items-center justify-center gap-2 border border-red-500/20"
                    title="Drop LLM from RAM"
                >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    "Drop RAM"
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
        <div class="fixed bottom-10 right-10 z-[9999]" style="pointer-events: none;">
            <Show when=move || show.get()>
                <button
                    on:click=scroll_to_top
                    class="p-4 bg-obsidian-accent text-white rounded-full shadow-[0_0_20px_rgba(92,124,250,0.6)] hover:bg-obsidian-accent/80 hover:scale-110 active:scale-95 transition-all duration-300 group flex items-center justify-center border border-white/20"
                    style="pointer-events: auto;"
                    title="Back to Top"
                >
                    <div class="flex flex-col items-center">
                        <svg class="w-6 h-6 group-hover:-translate-y-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 15l7-7 7 7" />
                        </svg>
                        <span class="text-[8px] font-bold mt-1 uppercase tracking-tighter">"TOP"</span>
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
            <div class="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" on:click=move |_| on_close.run(())>
                <div class="bg-obsidian-sidebar border border-white/10 rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200" on:click=|ev| ev.stop_propagation()>
                    <div class="p-6 border-b border-white/5 flex justify-between items-center">
                        <div class="flex items-center gap-2">
                            <svg class="w-5 h-5 text-obsidian-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <h2 class="text-xl font-bold text-obsidian-heading">"About arXiv Daily"</h2>
                        </div>
                        <button on:click=move |_| on_close.run(()) class="text-obsidian-text/40 hover:text-white transition-colors">"✕"</button>
                    </div>
                    <div class="p-8 space-y-6">
                        <div class="flex flex-col items-center text-center space-y-4 mb-4">
                            <div class="relative group/logo">
                                <div class="absolute -inset-1 bg-gradient-to-r from-obsidian-accent to-purple-500 rounded-2xl blur opacity-25 group-hover/logo:opacity-50 transition duration-1000 group-hover/logo:duration-200"></div>
                                <img src="/logo.svg" alt="Logo" class="relative w-20 h-20 rounded-2xl shadow-xl bg-obsidian-sidebar" />
                            </div>
                            <div>
                                <h3 class="text-2xl font-black text-white italic tracking-tighter">"arXiv" <span class="text-obsidian-accent">"Daily"</span></h3>
                                <p class="text-xs text-obsidian-text/40 font-bold uppercase tracking-widest mt-1">"Research Discovery Engine"</p>
                            </div>
                        </div>

                        <div class="bg-obsidian-bg rounded-xl border border-white/5 overflow-hidden">
                            <Transition fallback=move || view! { <div class="p-4 text-center animate-pulse text-obsidian-text/20">"Retrieving build metadata..."</div> }>
                                {move || version_resource.get().map(|res| {
                                    match res {
                                        Ok(info) => view! {
                                            <div class="max-h-[65vh] overflow-y-auto custom-scrollbar divide-y divide-white/5">
                                                <div class="p-6 space-y-4">
                                                    <h4 class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-accent">"Software Info"</h4>
                                                    <div class="space-y-3 text-sm">
                                                        <div class="flex justify-between">
                                                            <span class="text-obsidian-text/40">"Version"</span>
                                                            <span class="text-obsidian-accent font-mono font-bold tracking-tight">{info.build_semver}</span>
                                                        </div>
                                                        <div class="flex justify-between">
                                                            <span class="text-obsidian-text/40">"Authors"</span>
                                                            <span class="text-obsidian-heading text-xs italic">{info.authors}</span>
                                                        </div>
                                                        <div class="flex flex-col gap-1">
                                                            <span class="text-obsidian-text/40 text-xs">"Repository"</span>
                                                            <span class="text-obsidian-heading font-mono text-[10px] truncate">{info.repository}</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div class="p-6 space-y-4">
                                                    <h4 class="text-[10px] font-black uppercase tracking-[0.2em] text-amber-400">"Build Metadata"</h4>
                                                    <div class="space-y-3 text-sm">
                                                        <div class="flex justify-between">
                                                            <span class="text-obsidian-text/40">"Built At"</span>
                                                            <span class="text-obsidian-heading font-mono text-xs">{info.build_timestamp}</span>
                                                        </div>
                                                        <div class="flex justify-between">
                                                            <span class="text-obsidian-text/40">"Optimization"</span>
                                                            <span class="text-amber-400/80 font-mono text-xs">"Level "{info.optimization}</span>
                                                        </div>
                                                        <div class="flex justify-between">
                                                            <span class="text-obsidian-text/40">"Debug Symbols"</span>
                                                            <span class="text-obsidian-heading font-mono text-xs">{info.debug_symbols}</span>
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
