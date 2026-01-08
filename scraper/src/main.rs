use std::time::Duration;

use chrono::DateTime;
use chrono::Utc;
use feed_rs::model::Entry;
use serde::Deserialize;
use serde::Serialize;
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

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS \
         papers (
            id TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            updated DATETIME,
            published DATETIME,
            summary TEXT,
            primary_category TEXT,
            categories TEXT,
            authors TEXT,
            pdf_link TEXT
    )",
    )
    .execute(&pool)
    .await?;

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

    // Start one transaction for all 2000 papers
    let mut tx = pool.begin().await?;

    for paper in papers {

        sqlx::query(
            "INSERT INTO papers (id, \
             url, title, updated, \
             published, summary, \
             primary_category, \
             categories, authors, \
             pdf_link)
        VALUES (?, ?, ?, ?, ?, ?, ?, \
             ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
        updated = excluded.updated,
        title = excluded.title,
        summary = excluded.summary,
        url = excluded.url",
        )
        .bind(&paper.id)
        .bind(&paper.url)
        .bind(&paper.title)
        .bind(paper.updated)
        .bind(paper.published)
        .bind(&paper.summary)
        .bind(&paper.primary_category)
        .bind(&paper.categories)
        .bind(&paper.authors)
        .bind(&paper.pdf_link)
        .execute(&mut *tx) // Execute inside the transaction
        .await?;
    }

    // Commit tells SQLite to actually save to the .db file
    tx.commit().await?;

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
