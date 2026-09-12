import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- load your already-tagged dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("tagged_divorce_cases.csv")
    return df

df = load_data()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# --- build the TF-IDF + similarity engine once, cached ---
@st.cache_resource
def build_engine(facts_series):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(facts_series)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_engine(df["cleaned_facts"])

# --- page layout ---
st.title("Divorce Case Recommender")
st.write("Enter case facts below, or filter by ground of divorce, to find similar past cases.")

# structured filter (sidebar)
st.sidebar.header("Filter (optional)")
grounds = ["All"] + sorted(df["ground_of_divorce"].unique().tolist())
selected_ground = st.sidebar.selectbox("Ground of divorce", grounds)

# free-text query
query = st.text_area("Describe the case facts:", height=120,
                      placeholder="e.g. husband repeatedly abusive, wife seeks divorce on grounds of cruelty")

top_n = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Find similar cases"):
    working_df = df.copy()

    # apply structured filter first
    if selected_ground != "All":
        working_df = working_df[working_df["ground_of_divorce"].str.contains(selected_ground, na=False)]

    if len(working_df) == 0:
        st.warning("No cases match that filter.")
    elif query.strip() == "":
        st.info("Showing filtered cases (no text query entered):")
        st.dataframe(working_df[["case_title", "ground_of_divorce", "final_verdict"]])
    else:
        # vectorize the query using the SAME vocabulary
        cleaned_query = clean_text(query)
        query_vector = vectorizer.transform([cleaned_query])

        # only compare against the filtered subset
        subset_indices = working_df.index.tolist()
        subset_matrix = tfidf_matrix[subset_indices]

        scores = cosine_similarity(query_vector, subset_matrix)[0]
        working_df = working_df.copy()
        working_df["similarity"] = scores
        results = working_df.sort_values("similarity", ascending=False).head(top_n)

        st.subheader(f"Top {len(results)} similar cases")
        for _, row in results.iterrows():
            with st.expander(f"{row['case_title']}  —  similarity: {row['similarity']:.2f}"):
                st.write(f"**Ground:** {row['ground_of_divorce']}")
                st.write(f"**Verdict:** {row['final_verdict']}")
                st.write(f"**Facts:** {row['facts_summary']}")