use feed_rs::parser;
use reqwest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = "http://export.arxiv.org/api/query?search_query=cat:hep-th&start=2001&max_results=5&sortBy=submittedDate&sortOrder=descending";
    let client = reqwest::Client::new();
    let response = client.get(url).send().await?.bytes().await?;
    let feed = parser::parse(&response[..])?;
    for entry in feed.entries {
        println!("ID: {}", entry.id);
    }
    Ok(())
}
