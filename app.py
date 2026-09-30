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

st.set_page_config(
    page_title="Legal Judgement Recommendation System",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
.result-card {
    background-color: #f8f9fb;
    color: #1a1a1a;
    border-left: 4px solid #4a5b8c;
    padding: 14px 18px;
    border-radius: 6px;
    margin-bottom: 12px;
}
.result-card b, .result-card strong { color: #1a1a1a; }
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

# domain-specific words that appear in almost every legal document and
# carry no distinguishing meaning for OUR similarity task
LEGAL_BOILERPLATE = {
    "court", "petitioner", "respondent", "appellant", "case", "section",
    "act", "high", "supreme", "judgment", "judgement", "order", "party",
    "parties", "learned", "filed", "hon", "vs", "versus", "civil", "appeal",
    "counsel", "matter", "present", "instant", "said"
}

stop_words = set(stopwords.words('english')) | LEGAL_BOILERPLATE

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

@st.cache_resource
def build_engine(facts_series):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(facts_series)
    return vectorizer, tfidf_matrix

# NOTE: cleaning changed above, so the cached "cleaned_facts" column from
# your CSV may be stale. We re-clean from facts_summary fresh, live, to
# make sure the new boilerplate removal actually applies.
df["cleaned_facts"] = df["facts_summary"].apply(clean_text)

vectorizer, tfidf_matrix = build_engine(df["cleaned_facts"])
feature_names = vectorizer.get_feature_names_out()

def get_matched_keywords(query_vector, case_vector, top_k=5):
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

tab_recommender, tab_about = st.tabs(["🔍 Recommender", "ℹ️ About this project"])

with tab_recommender:
    st.title("⚖️ Legal Judgement Recommendation System")
    st.caption("Enter case facts, or filter by ground of divorce, to find similar past legal judgements with their outcomes.")

    st.sidebar.header("Filter (optional)")
    grounds = ["All"] + sorted(df["ground_of_divorce"].unique().tolist())
    selected_ground = st.sidebar.selectbox("Ground of divorce", grounds)

    with st.sidebar.expander("⚙️ Advanced settings"):
        similarity_threshold = st.slider(
            "Minimum similarity to count as relevant", 0.0, 0.5, 0.12, 0.01
        )
        require_keyword_overlap = st.checkbox(
            "Require at least one matched keyword", value=True
        )

    query = st.text_area("Describe the case facts:", height=120,
                          placeholder="e.g. husband repeatedly abusive, wife seeks divorce on grounds of cruelty")
    top_n = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("Find similar judgements", type="primary"):
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

            # first filter: similarity threshold
            relevant = working_df[working_df["similarity"] >= similarity_threshold]

            # second filter: must share at least one real matched keyword
            if require_keyword_overlap and len(relevant) > 0:
                keep_indices = []
                for idx in relevant.index:
                    case_vector = tfidf_matrix[idx]
                    matched = get_matched_keywords(query_vector, case_vector, top_k=1)
                    if len(matched) > 0:
                        keep_indices.append(idx)
                relevant = relevant.loc[keep_indices]

            results = relevant.sort_values("similarity", ascending=False).head(top_n)

            if len(results) == 0:
                best_score = working_df['similarity'].max()
                st.warning(
                    f"No sufficiently relevant cases found (highest match was "
                    f"{best_score:.2f}). Try rephrasing with more specific legal "
                    f"facts (e.g. grounds, circumstances, relevant terms)."
                )
            else:
                st.subheader(f"Top {len(results)} similar cases")

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

                    with st.expander("View full case text"):
                        st.text_area("Full judgement text", row['full_text'], height=300, label_visibility="collapsed")

with tab_about:
    st.title("About this project")
    st.markdown("""
    ### Project Exhibition 1 — Legal Judgement Recommendation System

    **Approach:** Content-based filtering using TF-IDF vectorization (unigrams + bigrams)
    and cosine similarity, combined with a structured filter layer (ground of divorce, verdict),
    a minimum relevance threshold, and a matched-keyword safety check to avoid recommending
    unrelated cases.

    **Dataset:** A curated, manually verified set of Indian divorce/matrimonial court judgements.

    **Pipeline:**
    1. Data collection and manual verification
    2. Ground-of-divorce tagging (keyword-based)
    3. Text preprocessing (cleaning, legal-boilerplate + stopword removal, lemmatization)
    4. TF-IDF vectorization (unigrams + bigrams)
    5. Cosine similarity search engine with relevance thresholding and keyword-overlap verification
    6. Streamlit web interface

    **Team:** [add your 6 team members' names here]
    """)

    st.subheader("Dataset overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total judgements", len(df))
    col2.metric("Unique grounds", df["ground_of_divorce"].nunique())
    col3.metric("Verdict types", df["final_verdict"].nunique())

    st.dataframe(df["ground_of_divorce"].value_counts().rename("count"), use_container_width=True)
