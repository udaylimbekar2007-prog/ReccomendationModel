# Divorce Case Recommender — Exhibition Report Data

## 1. Corpus size and label counts (matches the "About" tab metrics)

- **Total cases:** 56
- **Unique grounds of divorce:** 8
- **Verdict types:** 5

**Grounds of divorce breakdown:**

| Ground | Count |
|---|---|
| Unspecified | 36 |
| Cruelty | 8 |
| Mutual Consent | 3 |
| Irretrievable Breakdown | 3 |
| Cruelty, Desertion | 3 |
| Cruelty, Irretrievable Breakdown | 1 |
| Mutual Consent, Irretrievable Breakdown | 1 |
| Desertion | 1 |

**Verdict breakdown:**

| Verdict | Count |
|---|---|
| Allowed | 38 |
| Dismissed | 8 |
| Disposed | 6 |
| Unknown | 3 |
| Quashed | 1 |

Note: 36/56 cases (64%) are tagged "Unspecified" for ground of divorce — worth flagging in your write-up as a limitation of the keyword-based tagging step, and a natural place to say "future work: improve ground tagging."

## 2. Example query with top-5 recommendations

**Query entered:** *"husband repeatedly abusive, wife seeks divorce on grounds of cruelty and mental harassment"*

| Case | Ground | Verdict | Similarity | Matched keywords |
|---|---|---|---|---|
| Raj Talreja vs Kavita Talreja (24 Apr 2017) | Cruelty | Allowed | 0.180 | wife, cruelty |
| Suman Singh vs Sanjay Singh (8 Mar 2017) | Cruelty | Allowed | 0.157 | cruelty, divorce, mental |
| Narendra vs K Meena (6 Oct 2016) | Unspecified | Allowed | 0.148 | wife, divorce, husband |
| Jalendra Padhiary vs Pragati Chhotray (17 Apr 2018) | Cruelty, Desertion | Allowed | 0.148 | wife, cruelty, husband |
| Amutha vs A R Subramanian (19 Dec 2024) | Unspecified | Dismissed | 0.108 | husband, divorce |

This is a real run of the actual TF-IDF + cosine-similarity pipeline in `app.py` against `tagged_divorce_cases.csv` (see caveat below).

## 3. Spot-check evaluation — precision@5

Four representative queries, each aimed at a specific ground of divorce, checked against whether the top-5 retrieved cases' `ground_of_divorce` field matched the intended theme:

| Query theme | Precision@5 | Top-5 verdicts |
|---|---|---|
| Cruelty | 3/5 = 0.60 | Allowed, Allowed, Allowed, Allowed, Dismissed |
| Mutual Consent | 2/5 = 0.40 | Allowed, Allowed, Allowed, Dismissed, Disposed |
| Desertion | 3/5 = 0.60 | Allowed, Allowed, Disposed, Allowed, Allowed |
| Irretrievable Breakdown | 2/5 = 0.40 | Allowed, Allowed, Allowed, Allowed, Allowed |

**Overall precision@5 across the 4 queries: 10/20 = 0.50**

Reasonable, honest framing for the report: half of the top-5 results for a themed query carry a `ground_of_divorce` tag matching that theme. Given that 64% of cases are tagged "Unspecified," this actually undercounts true relevance — several "Unspecified" hits were topically on-point (shared vocabulary like "cruelty," "husband," "wife") but just weren't ground-tagged. Worth noting this nuance rather than the bare number alone.
