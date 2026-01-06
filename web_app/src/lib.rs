use leptos::prelude::*;
use leptos_meta::*;
use leptos_router::components::*;
use leptos_router::*;
use shared::Paper;


#[server(GetPapers, "/api")]
pub async fn get_papers(query: String) -> Result<Vec<Paper>, ServerFnError> {
    use sqlx::sqlite::SqlitePool;
    use toml;
    use serde::Deserialize;
    use chrono::{DateTime, Utc, NaiveDateTime};

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

    let db_url = format!("sqlite:{}?mode=ro", config.database.path);
    let pool = SqlitePool::connect(&db_url).await
        .map_err(|e| ServerFnError::new(format!("Db connection error: {}", e)))?;

    let rows = if query.is_empty() {
        sqlx::query(
            "SELECT id, url, title, updated, published, summary, primary_category, categories, authors, pdf_link FROM papers ORDER BY published DESC LIMIT 100"
        )
        .fetch_all(&pool)
        .await
        .map_err(|e| ServerFnError::new(format!("Query error: {}", e)))?
    } else {
        let search_pattern = format!("%{}%", query);
        sqlx::query(
            "SELECT id, url, title, updated, published, summary, primary_category, categories, authors, pdf_link FROM papers WHERE title LIKE ? OR summary LIKE ? ORDER BY published DESC LIMIT 100"
        )
        .bind(&search_pattern)
        .bind(&search_pattern)
        .fetch_all(&pool)
        .await
        .map_err(|e| ServerFnError::new(format!("Query error: {}", e)))?
    };

    use rayon::prelude::*;
    use sqlx::Row;
    let papers: Vec<Paper> = rows.into_par_iter().map(|row| {
        let updated_naive: NaiveDateTime = row.get("updated");
        let published_naive: NaiveDateTime = row.get("published");
        
        Paper {
            id: row.get("id"),
            url: row.get::<Option<String>, _>("url").unwrap_or_default(),
            title: row.get::<Option<String>, _>("title").unwrap_or_default(),
            updated: updated_naive.and_utc().timestamp(),
            published: published_naive.and_utc().timestamp(),
            summary: row.get::<Option<String>, _>("summary").unwrap_or_default(),
            primary_category: row.get::<Option<String>, _>("primary_category").unwrap_or_default(),
            categories: row.get::<Option<String>, _>("categories").unwrap_or_default(),
            authors: row.get::<Option<String>, _>("authors").unwrap_or_default(),
            pdf_link: row.get("pdf_link"),
        }
    }).collect();

    Ok(papers)
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
                <Stylesheet id="leptos" href="/pkg/web_app.css"/>
                <Title text="arXiv Dashboard"/>
                <Meta name="description" content="Personalized research discovery platform"/>
            </head>
            <body>
                <App/>
            </body>
        </html>
    }
}

#[component]
pub fn App() -> impl IntoView {
    view! {
        <Router>
            <main class="min-h-screen bg-obsidian-bg text-obsidian-text font-sans selection:bg-obsidian-accent/30">
                <Routes fallback=|| "Not Found">
                    <Route path=path!("") view=Dashboard/>
                </Routes>
            </main>
        </Router>
    }
}

#[component]
fn Dashboard() -> impl IntoView {
    let (input_val, set_input_val) = create_signal("".to_string());
    let (debounce_query, set_debounce_query) = create_signal("".to_string());
    
    // Debounce logic
    create_effect(move |_| {
        let new_val = input_val.get();
        let timeout = set_timeout_with_handle(move || {
            set_debounce_query.set(new_val);
        }, std::time::Duration::from_millis(300));
        
        move || {
            if let Ok(timeout) = timeout {
                timeout.clear();
            }
        }
    });

    let papers = Resource::new(
        move || debounce_query.get(),
        |q| async move { get_papers(q).await.unwrap_or_default() }
    );

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
                        prop:value=input_val
                    />
                </div>
            </header>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 gap-8">
                <Transition fallback=move || view! { 
                    <div class="col-span-full py-32 flex flex-col items-center justify-center space-y-4">
                        <div class="w-12 h-12 border-4 border-obsidian-accent/20 border-t-obsidian-accent rounded-full animate-spin"></div>
                        <p class="text-obsidian-text/40 font-medium animate-pulse">"Indexing archive databases..."</p>
                    </div> 
                }>
                    {move || papers.get().map(|data: Vec<Paper>| {
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
                    })}
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
    let (expanded, set_expanded) = create_signal(false);
    
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

#[cfg(feature = "hydrate")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn hydrate() {
    _ = console_log::init_with_level(log::Level::Debug);
    console_error_panic_hook::set_once();
    leptos::mount_to_body(App);
}
