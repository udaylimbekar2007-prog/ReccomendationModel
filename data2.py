import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# one-time downloads (only needs to run once ever)
nltk.download('stopwords')
nltk.download('wordnet')

# --- Step 1: load your clean dataset ---
df = pd.read_csv("clean_divorce_cases.csv")
print("Loaded cases:", len(df))

# --- Step 2: auto-tag ground of divorce based on keywords in facts_summary ---
ground_keywords = {
    "Cruelty": ["cruelty", "harassment", "abuse", "dowry"],
    "Desertion": ["desertion", "abandonment", "deserted"],
    "Adultery": ["adultery", "extramarital", "infidelity"],
    "Mutual Consent": ["mutual consent", "13-b", "13b"],
    "Irretrievable Breakdown": ["irretrievable breakdown", "irretrievably broken"],
}

def tag_ground(text):
    text = str(text).lower()
    tags = []
    for ground, keywords in ground_keywords.items():
        if any(kw in text for kw in keywords):
            tags.append(ground)
    return ", ".join(tags) if tags else "Unspecified"

df["ground_of_divorce"] = df["facts_summary"].apply(tag_ground)
print("\nGround distribution:")
print(df["ground_of_divorce"].value_counts())

# --- Step 3: clean the text ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)          # remove punctuation/numbers
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df["cleaned_facts"] = df["facts_summary"].apply(clean_text)

# --- Step 4: TF-IDF vectorization ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["cleaned_facts"])
print("\nTF-IDF matrix shape:", tfidf_matrix.shape)   # (num_cases, vocabulary_size)

# --- Step 5: cosine similarity matrix (every case vs every other case) ---
similarity_matrix = cosine_similarity(tfidf_matrix)
print("Similarity matrix shape:", similarity_matrix.shape)

# --- Step 6: recommend function ---
def recommend(case_index, top_n=5):
    scores = list(enumerate(similarity_matrix[case_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != case_index][:top_n]   # exclude itself

    print(f"\nQuery case: {df.iloc[case_index]['case_title']}")
    print(f"Ground: {df.iloc[case_index]['ground_of_divorce']}\n")
    print(f"Top {top_n} similar cases:\n")
    for idx, score in scores:
        print(f"  {df.iloc[idx]['case_title']}")
        print(f"    Ground: {df.iloc[idx]['ground_of_divorce']} | Verdict: {df.iloc[idx]['final_verdict']} | Similarity: {score:.3f}\n")

# --- Step 7: save everything so the app layer can use it later ---
df.to_csv("tagged_divorce_cases.csv", index=False)
print("Saved tagged_divorce_cases.csv")

# quick test — try recommending for the first case
recommend(0, top_n=5)