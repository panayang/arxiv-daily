use leptos::prelude::*;
use leptos_meta::*;
use leptos_router::components::*;
use leptos_router::*;
use shared::{Paper, Category};

#[server(GetPapers, "/api")]
pub async fn get_papers(query: String, category: String, date: String, end_date: String) -> Result<Vec<Paper>, ServerFnError> {
    use chrono::{DateTime, Utc};
    use sqlx::sqlite::SqlitePool;

    let db_path = get_db_path().await?;
    let db_url = format!("sqlite:{}?mode=ro", db_path);
    let pool = SqlitePool::connect(&db_url)
        .await
        .map_err(|e| ServerFnError::new(format!("Db connection error: {}", e)))?;

    log::info!("Searching papers: query='{}', cat='{}', date='{}'", query, category, date);

    let mut sql = "SELECT id, url, title, updated, published, summary, primary_category, categories, authors, pdf_link FROM papers WHERE 1=1".to_string();
    
    if !query.is_empty() {
        sql.push_str(" AND (title LIKE ? OR summary LIKE ?)");
    }
    if !category.is_empty() && category != "all" {
        sql.push_str(" AND primary_category = ?");
    }
    if !date.is_empty() {
        if !end_date.is_empty() {
            sql.push_str(" AND date(published) BETWEEN ? AND ?");
        } else {
            sql.push_str(" AND date(published) = ?");
        }
    }
    
    sql.push_str(" ORDER BY published DESC LIMIT 50");

    let mut q = sqlx::query(&sql);
    
    if !query.is_empty() {
        let pattern = format!("%{}%", query);
        q = q.bind(pattern.clone()).bind(pattern);
    }
    if !category.is_empty() && category != "all" {
        q = q.bind(category);
    }
    if !date.is_empty() {
        if !end_date.is_empty() {
            q = q.bind(date).bind(end_date);
        } else {
            q = q.bind(date);
        }
    }

    let rows = q.fetch_all(&pool).await
        .map_err(|e| {
            log::error!("Database query error: {}", e);
            ServerFnError::new(format!("Query error: {}", e))
        })?;

    log::info!("Found {} papers matching criteria", rows.len());

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

    Ok(papers)
}

#[server(GetConfig, "/api")]
pub async fn get_config() -> Result<String, ServerFnError> {
    let content = std::fs::read_to_string("config.toml")
        .or_else(|_| std::fs::read_to_string("../config.toml"))
        .map_err(|e| ServerFnError::new(format!("Failed to read config: {}", e)))?;
    Ok(content)
}

#[server(SaveConfig, "/api")]
pub async fn save_config(content: String) -> Result<(), ServerFnError> {
    let path = if std::path::Path::new("config.toml").exists() {
        "config.toml"
    } else {
        "../config.toml"
    };
    std::fs::write(path, content)
        .map_err(|e| ServerFnError::new(format!("Failed to write config: {}", e)))?;
    Ok(())
}

#[server(FetchNewArticles, "/api")]
pub async fn fetch_new_articles(category: String, start_date: String, end_date: String) -> Result<usize, ServerFnError> {
    use sqlx::sqlite::SqlitePool;
    use chrono::Utc;
    use feed_rs::model::Entry;

    let db_path = get_db_path().await?;
    let db_url = format!("sqlite:{}?mode=rwc", db_path);
    let pool = SqlitePool::connect(&db_url).await
        .map_err(|e| ServerFnError::new(format!("Db connection error: {}", e)))?;

    use serde::Deserialize;
    use toml;
    #[derive(Deserialize)]
    struct ArxivConfig { 
        category: String,
        #[serde(default)]
        start: i32,
        #[serde(default = "default_max_results")]
        max_results: i32,
    }
    fn default_max_results() -> i32 { 50 }
    #[derive(Deserialize)]
    struct Config { arxiv: ArxivConfig }
    
    let config_content = std::fs::read_to_string("config.toml")
        .or_else(|_| std::fs::read_to_string("../config.toml"))
        .map_err(|e| ServerFnError::new(format!("Failed to read config: {}", e)))?;
    let config: Config = toml::from_str(&config_content)
        .map_err(|e| ServerFnError::new(format!("Failed to parse config: {}", e)))?;

    let target_category = if category == "all" { config.arxiv.category } else { category };

    let url = if !end_date.is_empty() {
        let s = if start_date.is_empty() { &end_date } else { &start_date };
        let s_fmt = s.replace("-", "") + "0600";
        let e_fmt = end_date.replace("-", "") + "0600";
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
    let response = client.get(url).send().await
        .map_err(|e| ServerFnError::new(format!("ArXiv request failed: {}", e)))?;

    let status = response.status();
    let bytes = response.bytes().await
        .map_err(|e| ServerFnError::new(format!("Failed to read body: {}", e)))?;

    if !status.is_success() {
        let err_msg = String::from_utf8_lossy(&bytes);
        return Err(ServerFnError::new(format!("ArXiv API error ({}): {}", status, err_msg)));
    }

    let feed = feed_rs::parser::parse(&bytes[..])
        .map_err(|e| ServerFnError::new(format!("Failed to parse feed: {}", e)))?;
    
    let count = feed.entries.len();
    
    let papers: Vec<Paper> = feed.entries.into_iter().map(|entry: Entry| {
        let authors: Vec<String> = entry.authors.iter().map(|a| a.name.clone()).collect();
        let pdf_link = entry.links.iter()
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
            updated: entry.updated.map(|d| d.timestamp()).unwrap_or_else(|| Utc::now().timestamp()),
            published: entry.published.map(|d| d.timestamp()).unwrap_or_else(|| Utc::now().timestamp()),
            summary: entry.summary.map(|s| s.content).unwrap_or_default(),
            primary_category: entry.categories.first().map(|c| c.term.clone()).unwrap_or_default(),
            categories: entry.categories.iter().map(|c| c.term.clone()).collect::<Vec<_>>().join(","),
            authors: serde_json::to_string(&authors).unwrap_or_default(),
            pdf_link,
        }
    }).collect();

    let mut tx = pool.begin().await
        .map_err(|e| ServerFnError::new(format!("Transaction start error: {}", e)))?;
    
    // Ensure table exists
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS papers (id TEXT PRIMARY KEY, url TEXT, title TEXT, updated DATETIME, published DATETIME, summary TEXT, primary_category TEXT, categories TEXT, authors TEXT, pdf_link TEXT)"
    ).execute(&pool).await.ok();

    for paper in papers {
        sqlx::query(
            "INSERT INTO papers (id, url, title, updated, published, summary, primary_category, categories, authors, pdf_link)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
        updated = excluded.updated,
        title = excluded.title,
        summary = excluded.summary,
        url = excluded.url"
        )
        .bind(&paper.id)
        .bind(&paper.url)
        .bind(&paper.title)
        .bind(chrono::NaiveDateTime::from_timestamp_opt(paper.updated, 0).unwrap_or_default())
        .bind(chrono::NaiveDateTime::from_timestamp_opt(paper.published, 0).unwrap_or_default())
        .bind(&paper.summary)
        .bind(&paper.primary_category)
        .bind(&paper.categories)
        .bind(&paper.authors)
        .bind(&paper.pdf_link)
        .execute(&mut *tx)
        .await
        .map_err(|e| ServerFnError::new(format!("Insert error: {}", e)))?;
    }

    tx.commit().await
        .map_err(|e| ServerFnError::new(format!("Commit error: {}", e)))?;

    Ok(count)
}

#[cfg(feature = "ssr")]
async fn get_db_path() -> Result<String, ServerFnError> {
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

    let config_content = std::fs::read_to_string("config.toml")
        .or_else(|_| std::fs::read_to_string("../config.toml"))
        .map_err(|e| ServerFnError::new(format!("Failed to read config: {}", e)))?;
    let config: Config = toml::from_str(&config_content)
        .map_err(|e| ServerFnError::new(format!("Failed to parse config: {}", e)))?;
    Ok(config.database.path)
}

pub fn shell(options: LeptosOptions) -> impl IntoView {
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
    let (input_val, set_input_val) = signal("".to_string());
    let (selected_category, set_selected_category) = signal("all".to_string());
    let (date_filter, set_date_filter) = signal("".to_string());
    let (end_date_filter, set_end_date_filter) = signal("".to_string());
    let (show_config, set_show_config) = signal(false);

    #[derive(Clone, Default, PartialEq)]
    struct SearchParams {
        query: String,
        category: String,
        date: String,
        end_date: String,
    }

    let (trigger_search, set_trigger_search) = signal(SearchParams {
        query: "".to_string(),
        category: "all".to_string(),
        date: "".to_string(),
        end_date: "".to_string(),
    });

    let papers = Resource::new(
        move || trigger_search.get(),
        |params| async move { get_papers(params.query, params.category, params.date, params.end_date).await },
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
            if let Ok(timeout) = timeout {
                timeout.clear();
            }
        }
    });

    // Auto-search on category/date change
    Effect::new(move |_| {
        let category = selected_category.get();
        let date = date_filter.get();
        let end_date = end_date_filter.get();
        set_trigger_search.update(|p| {
            p.category = category;
            p.date = date;
            p.end_date = end_date;
        });
    });

    let fetch_action = Action::new(move |(cat, start, end): &(String, String, String)| {
        let cat = cat.clone();
        let start = start.clone();
        let end = end.clone();
        async move {
            let res = fetch_new_articles(cat, start, end).await;
            if res.is_ok() {
                set_trigger_search.update(|p| {
                    p.query = "".to_string();
                    p.category = "all".to_string();
                    p.date = "".to_string();
                    p.end_date = "".to_string();
                });
            }
            res
        }
    });

    let on_search = move |_| {
        set_trigger_search.set(SearchParams {
            query: input_val.get(),
            category: selected_category.get(),
            date: date_filter.get(),
            end_date: end_date_filter.get(),
        });
    };

    let on_reset = move |_| {
        set_input_val.set("".to_string());
        set_selected_category.set("all".to_string());
        set_date_filter.set("".to_string());
        set_end_date_filter.set("".to_string());
        set_trigger_search.set(SearchParams {
            query: "".to_string(),
            category: "all".to_string(),
            date: "".to_string(),
            end_date: "".to_string(),
        });
    };
    view! {
        <div class="max-w-7xl mx-auto px-4 py-8 md:px-8 md:py-12 space-y-12">
            <header class="flex flex-col md:flex-row md:items-end justify-between gap-6 border-b border-white/10 pb-10">
                <div class="space-y-2">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 bg-obsidian-accent rounded-lg flex items-center justify-center shadow-lg shadow-obsidian-accent/20">
                            <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                            </svg>
                        </div>
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

            <FilterBar
                selected_category=selected_category.into()
                set_selected_category
                date_filter=date_filter.into()
                set_date_filter
                on_fetch=Callback::new(move |_| {
                    fetch_action.dispatch((selected_category.get(), date_filter.get(), end_date_filter.get()));
                })
                on_search=Callback::new(on_search)
                on_reset=Callback::new(on_reset)
                on_edit_config=Callback::new(move |_| set_show_config.set(true))
                end_date_filter=end_date_filter.into()
                set_end_date_filter
            />

            <ConfigModal show=show_config.into() on_close=Callback::new(move |_| set_show_config.set(false))/>

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
                                let start_str = date_filter.get();
                                let end_str = end_date_filter.get();
                                
                                let data = if start_str.is_empty() {
                                    current_papers
                                } else if end_str.is_empty() {
                                    // Single day filter
                                    current_papers.into_iter().filter(|p| {
                                        p.published_date().format("%Y-%m-%d").to_string() == start_str
                                    }).collect()
                                } else {
                                    // Range filter
                                    current_papers.into_iter().filter(|p| {
                                        let p_date = p.published_date().format("%Y-%m-%d").to_string();
                                        p_date >= start_str && p_date <= end_str
                                    }).collect()
                                };

                                if data.is_empty() {
                                    view! {
                                        <div class="col-span-full py-32 text-center space-y-4">
                                            <div class="text-6xl text-obsidian-text/10 italic font-bold">"∅"</div>
                                            <p class="text-obsidian-text/40 text-lg">"No papers match your search criteria."</p>
                                        </div>
                                    }.into_any()
                                } else {
                                    view! {
                                        <For
                                            each=move || data.clone()
                                            key=|paper| paper.id.clone()
                                            children=|paper| {
                                                view! { <PaperCard paper=paper/> }
                                            }
                                        />
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

            <footer class="pt-20 pb-10 border-t border-white/5 text-center">
                <p class="text-xs text-obsidian-text/20 uppercase tracking-[0.2em] font-bold">
                    "Powered by Rust • Leptos • Bitcode"
                </p>
            </footer>
        </div>
    }
}

#[component]
fn PaperCard(paper: Paper) -> impl IntoView {
    let (expanded, set_expanded) = signal(false);

    let authors = paper.authors_list();
    let display_authors = authors.join(", ");
    let title = paper.title.clone();
    let url = paper.url.clone();
    let summary = paper.summary.clone();
    let pdf_link = paper.pdf_link.clone();
    let published_str = paper.published_date().format("%b %d, %Y").to_string();
    let category_name = paper.primary_category_name();

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
                    view! {}.into_any()
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
fn ConfigModal(show: Signal<bool>, on_close: Callback<()>) -> impl IntoView {
    let config_resource = Resource::new(move || show.get(), |_| async move { get_config().await });
    let save_action = Action::new(|content: &String| {
        let content = content.clone();
        async move { save_config(content).await }
    });

    let (content, set_content) = signal("".to_string());

    Effect::new(move |_| {
        if let Some(Ok(c)) = config_resource.get() {
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
fn FilterBar(
    selected_category: Signal<String>,
    set_selected_category: WriteSignal<String>,
    date_filter: Signal<String>,
    set_date_filter: WriteSignal<String>,
    end_date_filter: Signal<String>,
    set_end_date_filter: WriteSignal<String>,
    on_fetch: Callback<()>,
    on_search: Callback<()>,
    on_reset: Callback<()>,
    on_edit_config: Callback<()>
) -> impl IntoView {
    let (category_search, set_category_search) = signal("".to_string());
    let (is_open, set_is_open) = signal(false);

    let filtered_categories = Memo::new(move |_| {
        let search = category_search.get().to_lowercase();
        let mut base = vec![("all", "All Categories")];
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
    });

    let current_category_name = move || {
        let code = selected_category.get();
        if code == "all" {
            "All Categories"
        } else {
            Category::ALL_CATEGORIES
                .iter()
                .find(|(c, _)| *c == code)
                .map(|(_, n)| *n)
                .unwrap_or("Unknown")
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
                    prop:value=date_filter
                />
            </div>

            <div class="flex flex-col gap-1.5 flex-1 min-w-[150px]">
                <label class="text-[10px] font-black uppercase tracking-[0.2em] text-obsidian-text/30 ml-1">"End Date"</label>
                <input
                    type="date"
                    class="bg-obsidian-bg border border-white/10 rounded-xl px-4 py-2 text-sm text-obsidian-heading focus:outline-none focus:ring-2 focus:ring-obsidian-accent/40"
                    on:input=move |ev| set_end_date_filter.set(event_target_value(&ev))
                    prop:value=end_date_filter
                />
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
                    class="flex-1 sm:flex-none h-11 px-6 bg-white/5 text-obsidian-text text-xs font-black uppercase tracking-widest rounded-xl hover:bg-white/10 transition-all border border-white/5 active:scale-95 flex items-center justify-center gap-2"
                >
                    <svg class="w-4 h-4 text-obsidian-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    "Fetch New"
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
            </div>
        </div>
    }
}

#[cfg(feature = "hydrate")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn hydrate() {
    _ = console_log::init_with_level(log::Level::Debug);
    console_error_panic_hook::set_once();
    leptos::mount::hydrate_body(App);
}
