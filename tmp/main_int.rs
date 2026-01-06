use sqlx::sqlite::SqlitePool;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create the physical file
    let db_url = "sqlite:arxiv_daily.db?mode=rwc";
    let pool = SqlitePool::connect(db_url).await?;

    // 2. Create the table using the dynamic query function
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            updated DATETIME,
            published DATETIME,
            summary TEXT,
            primary_category TEXT,
            categories TEXT,
            authors TEXT,
            pdf_link TEXT
    )"
    )
    .execute(&pool)
    .await?;

    println!("Database initialized successfully.");
    Ok(())
}
