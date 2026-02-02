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

//! This is the scraper for the arxiv-daily project.

#![allow(
    clippy::empty_line_after_outer_attr
)]

use chrono::Utc;
use feed_rs::model::Entry;
use serde::Deserialize;
use shared::Paper;
use sqlx::sqlite::SqlitePool;

#[derive(Debug, Deserialize)]

struct Config {
    arxiv: ArxivConfig,
    database: DatabaseConfig,
}

#[derive(Debug, Deserialize)]

struct ArxivConfig {
    category: String,
    start: u32,
    max_results: u32,
}

#[derive(Debug, Deserialize)]

struct DatabaseConfig {
    path: String,
}

#[tokio::main]

async fn main() -> Result<
    (),
    Box<dyn std::error::Error>,
> {

    // 0. Load Configuration
    let config_content =
        std::fs::read_to_string(
            "config.toml",
        )?;

    let config: Config =
        toml::from_str(
            &config_content,
        )?;

    // 1. Initialize Database
    // Ensure the parent directory exists
    if let Some(parent) =
        std::path::Path::new(
            &config.database.path,
        )
        .parent()
    {

        std::fs::create_dir_all(
            parent,
        )?;
    }

    let db_url = format!(
        "sqlite:{}?mode=rwc",
        config.database.path
    );

    let pool =
        SqlitePool::connect(&db_url)
            .await?;

    ensure_schema(&pool).await?;

    // 2. Fetch from arXiv
    // https://export.arxiv.org/api/query?search_query=submittedDate:[202401010600+TO+202401050600]
    println!(
        "Fetching from arXiv (cat:{}, \
         start:{}, max:{})...",
        config
            .arxiv
            .category,
        config.arxiv.start,
        config
            .arxiv
            .max_results
    );

    let url = format!(
        "http://export.arxiv.org/api/query?search_query=cat:{}&start={}&max_results={}&sortBy=submittedDate&sortOrder=descending",
        config.arxiv.category, config.arxiv.start, config.arxiv.max_results
    );

    let client = reqwest::Client::new();

    let response = client
        .get(url)
        .send()
        .await?
        .bytes()
        .await?;

    // 3. Parse and Transform
    let feed = feed_rs::parser::parse(
        &response[..],
    )?;

    println!(
        "Found {} entries. \
         Processing...",
        feed.entries.len()
    );

    // Use Rayon for CPU-bound transformation
    let papers: Vec<Paper> =
        tokio::task::spawn_blocking(
            move || {

                use rayon::prelude::*;

                feed.entries
                    .into_par_iter()
                    .map(
                        transform_entry,
                    )
                    .collect()
            },
        )
        .await?;

    // 4. Batch Save to SQLite
    println!(
        "Saving {} papers to \
         database...",
        papers.len()
    );

    if !papers.is_empty() {

        // Bulk insert using a single query for better performance
        let mut query_builder: sqlx::QueryBuilder<sqlx::Sqlite> = sqlx::QueryBuilder::new(
            "INSERT INTO papers (id, url, title, updated, published, summary, primary_category, categories, authors, pdf_link) "
        );

        query_builder.push_values(papers, |mut b, paper| {
            b.push_bind(paper.id)
                .push_bind(paper.url)
                .push_bind(paper.title)
                .push_bind(
                    chrono::DateTime::from_timestamp(paper.updated, 0).unwrap_or_default()
                )
                .push_bind(
                    chrono::DateTime::from_timestamp(paper.published, 0).unwrap_or_default()
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
            .await?;
    }

    println!(
        "Success! All papers saved."
    );

    println!(
        "Done! Check {}",
        config.database.path
    );

    Ok(())
}

fn transform_entry(
    entry: Entry
) -> Paper {

    let authors: Vec<String> = entry
        .authors
        .iter()
        .map(|a| a.name.clone())
        .collect();

    // Extract PDF link specifically
    let pdf_link = entry
        .links
        .iter()
        .find(|l| {

            l.media_type
                .as_deref()
                == Some(
                    "application/pdf",
                )
        })
        .map(|l| l.href.clone());

    let raw_id = entry.id.clone();

    let parsed_id = if let Some(pos) =
        raw_id.find("/abs/")
    {

        let s = &raw_id[pos + 5 ..];

        // Remove version suffix like v1
        if let Some(v_pos) =
            s.rfind('v')
        {

            if s[v_pos + 1 ..]
                .chars()
                .all(|c| {

                    c.is_ascii_digit()
                })
            {

                s[.. v_pos].to_string()
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
        title: entry
            .title
            .map(|t| t.content)
            .unwrap_or_default(),
        updated: entry
            .updated
            .map(|d| d.timestamp())
            .unwrap_or_else(|| {

                Utc::now().timestamp()
            }),
        published: entry
            .published
            .map(|d| d.timestamp())
            .unwrap_or_else(|| {

                Utc::now().timestamp()
            }),
        summary: entry
            .summary
            .map(|s| s.content)
            .unwrap_or_default(),
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
        authors: serde_json::to_string(
            &authors,
        )
        .unwrap_or_default(),
        pdf_link,
    }
}

async fn ensure_schema(
    pool: &SqlitePool
) -> Result<
    (),
    Box<dyn std::error::Error>,
> {

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
    .await?;

    // FTS5 Virtual Table for searching
    sqlx::query(
        "CREATE VIRTUAL TABLE IF NOT \
         EXISTS papers_fts USING \
         fts5(id UNINDEXED, title, \
         summary, content='papers', \
         content_rowid='rowid')",
    )
    .execute(pool)
    .await?;

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
    .await?;

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
    .await?;

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
    .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS \
         idx_papers_published ON \
         papers (published DESC)",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS \
         idx_papers_primary_category \
         ON papers (primary_category)",
    )
    .execute(pool)
    .await?;

    // Initial population of FTS if empty
    let row: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM \
         papers_fts",
    )
    .fetch_one(pool)
    .await
    .unwrap_or((0,));

    if row.0 == 0 {

        sqlx::query(
            "INSERT INTO \
             papers_fts(rowid, id, \
             title, summary) SELECT \
             rowid, id, title, \
             summary FROM papers",
        )
        .execute(pool)
        .await?;
    }

    Ok(())
}
