import streamlit as st
import pandas as pd
import httpx  # ✅ Used to call FastAPI backend

st.set_page_config(page_title="Research Supervisor AI", layout="wide")
st.title("🔍 Research Supervisor AI")
query = st.text_input("Enter a topic:", "")

def fetch_results(query):
    """Calls FastAPI backend to fetch research results."""
    with st.spinner(f"🔍 Researching '{query}'..."):
        response = httpx.get(f"http://127.0.0.1:8000/search/?query={query}")
        return response.json() if response.status_code == 200 else None

if st.button("Start Research") and query:
    data = fetch_results(query)

    if data:
        st.subheader("📄 Final Research Summary")
        st.markdown(data["final_response"])

        tab_titles = ["📢 Reddit", "🌍 Tavily", "📺 YouTube", "📖 Wikipedia", "📰 Hacker News", "🗞️ NewsAPI", "📄 Arxiv"]
        tabs = st.tabs(tab_titles)

        def display_results(source, results):
            if not results:
                st.warning(f"No results found for {source}.")
                return
            df = pd.DataFrame(results)
            if "title" in df.columns and "url" in df.columns:
                st.write(df[["title", "url"]].to_dict(orient="records"))
            else:
                st.write(df)

        for i, source in enumerate(tab_titles):
            with tabs[i]: display_results(source, data["raw_results"][source])
