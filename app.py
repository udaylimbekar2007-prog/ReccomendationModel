import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# --- page config (must be the first Streamlit command) ---
st.set_page_config(
    page_title="Divorce Case Recommender",
    page_icon="⚖️",
    layout="wide"
)

# --- light custom styling ---
st.markdown("""
<style>
.result-card {
    background-color: #f8f9fb;
    border-left: 4px solid #4a5b8c;
    padding: 14px 18px;
    border-radius: 6px;
    margin-bottom: 12px;
}
.verdict-granted { color: #1a7a3c; font-weight: 600; }
.verdict-denied { color: #b03434; font-weight: 600; }
.keyword-chip {
    display: inline-block;
    background-color: #e4e9f7;
    color: #2b3a67;
    padding: 2px 10px;
    border-radius: 12px;
    margin: 2px 4px 2px 0;
    font-size: 0.85em;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("tagged_divorce_cases.csv")

df = load_data()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

@st.cache_resource
def build_engine(facts_series):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(facts_series)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_engine(df["cleaned_facts"])
feature_names = vectorizer.get_feature_names_out()

def get_matched_keywords(query_vector, case_vector, top_k=5):
    """Return words that both the query and the case share, ranked by combined TF-IDF weight."""
    q = query_vector.toarray()[0]
    c = case_vector.toarray()[0]
    overlap_scores = q * c
    top_indices = overlap_scores.argsort()[::-1][:top_k]
    return [feature_names[i] for i in top_indices if overlap_scores[i] > 0]

def verdict_badge(verdict):
    v = str(verdict).lower()
    if "grant" in v:
        return f'<span class="verdict-granted">✅ {verdict}</span>'
    elif "den" in v or "dismiss" in v:
        return f'<span class="verdict-denied">❌ {verdict}</span>'
    return f"**{verdict}**"

# --- navigation ---
tab_recommender, tab_about = st.tabs(["🔍 Recommender", "ℹ️ About this project"])

with tab_recommender:
    st.title("⚖️ Divorce Case Recommender")
    st.caption("Enter case facts, or filter by ground of divorce, to find similar past cases with their outcomes.")

    st.sidebar.header("Filter (optional)")
    grounds = ["All"] + sorted(df["ground_of_divorce"].unique().tolist())
    selected_ground = st.sidebar.selectbox("Ground of divorce", grounds)

    query = st.text_area("Describe the case facts:", height=120,
                          placeholder="e.g. husband repeatedly abusive, wife seeks divorce on grounds of cruelty")
    top_n = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("Find similar cases", type="primary"):
        working_df = df.copy()

        if selected_ground != "All":
            working_df = working_df[working_df["ground_of_divorce"].str.contains(selected_ground, na=False)]

        if len(working_df) == 0:
            st.warning("No cases match that filter.")
        elif query.strip() == "":
            st.info("Showing filtered cases (no text query entered):")
            st.dataframe(working_df[["case_title", "ground_of_divorce", "final_verdict"]], use_container_width=True)
        else:
            cleaned_query = clean_text(query)
            query_vector = vectorizer.transform([cleaned_query])

            subset_indices = working_df.index.tolist()
            subset_matrix = tfidf_matrix[subset_indices]

            scores = cosine_similarity(query_vector, subset_matrix)[0]
            working_df = working_df.copy()
            working_df["similarity"] = scores
            results = working_df.sort_values("similarity", ascending=False).head(top_n)

            st.subheader(f"Top {len(results)} similar cases")

            # similarity chart across the results
            chart_data = results.set_index("case_title")["similarity"]
            st.bar_chart(chart_data)

            for idx, row in results.iterrows():
                case_vector = tfidf_matrix[idx]
                matched = get_matched_keywords(query_vector, case_vector)
                keyword_html = "".join([f'<span class="keyword-chip">{kw}</span>' for kw in matched]) or "<i>no strong keyword overlap</i>"

                st.markdown(f"""
                <div class="result-card">
                    <b>{row['case_title']}</b><br>
                    Ground: {row['ground_of_divorce']} &nbsp;|&nbsp; Verdict: {verdict_badge(row['final_verdict'])} &nbsp;|&nbsp; Similarity: {row['similarity']:.2f}
                    <br><br>
                    <b>Matched terms:</b><br>{keyword_html}
                </div>
                """, unsafe_allow_html=True)

                with st.expander("View full facts"):
                    st.write(row['facts_summary'])

with tab_about:
    st.title("About this project")
    st.markdown("""
    ### Project Exhibition 1 — Divorce Case Recommendation System

    **Approach:** Content-based filtering using TF-IDF vectorization and cosine similarity,
    combined with a structured filter layer (ground of divorce, verdict).

    **Dataset:** A curated, manually verified set of Indian divorce/matrimonial court judgments,
    filtered from a larger open legal case dataset.

    **Pipeline:**
    1. Data collection and manual verification
    2. Ground-of-divorce tagging (keyword-based)
    3. Text preprocessing (cleaning, stopword removal, lemmatization)
    4. TF-IDF vectorization
    5. Cosine similarity search engine
    6. Streamlit web interface

    **Team:** [add your 6 team members' names here]
    """)

    st.subheader("Dataset overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total cases", len(df))
    col2.metric("Unique grounds", df["ground_of_divorce"].nunique())
    col3.metric("Verdict types", df["final_verdict"].nunique())

    st.dataframe(df["ground_of_divorce"].value_counts().rename("count"), use_container_width=True)
        for idx, row in results.iterrows():
            with st.expander(f"{row['case_title']}  —  similarity: {row['similarity']:.2f}"):
                st.write(f"**Ground:** {row['ground_of_divorce']}")
                st.write(f"**Verdict:** {row['final_verdict']}")
                st.write(f"**Facts:** {row['facts_summary']}")
