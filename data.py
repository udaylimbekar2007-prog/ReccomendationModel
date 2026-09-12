import pandas as pd
import os

folder = r"C:\Users\udayl\Downloads\18984839"
file1 = os.path.join(folder, "Legal_Judgement_Predction(2016-2020).csv")
file2 = os.path.join(folder, "Legal_Judgement_Predction(2021-2025).csv")

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df = pd.concat([df1, df2], ignore_index=True)

print("Total cases loaded:", len(df))

# First, just see what case_type values exist
print("\nUnique case_type values:")
print(df["case_type"].value_counts())

# Now properly search text columns without truncation
keywords = ["hindu marriage act", "divorce", "irretrievable breakdown",
            "judicial separation", "mutual consent", "cruelty", "desertion"]

text_columns = ["facts_summary", "issues_raised", "acts_summary", "full_text", "case_type", "keywords"]
text_columns = [c for c in text_columns if c in df.columns]

mask = pd.Series(False, index=df.index)
for col in text_columns:
    col_lower = df[col].astype(str).str.lower()
    for kw in keywords:
        mask = mask | col_lower.str.contains(kw, na=False)

filtered = df[mask]
print(f"\nFound {len(filtered)} divorce-related cases out of {len(df)} total")

filtered.to_csv("divorce_cases_subset.csv", index=False)
print("Saved to divorce_cases_subset.csv")

family_law = df[df["case_type"] == "Family Law"]
print("Family Law cases:", len(family_law))

# how many of the 350 keyword matches are ALSO tagged Family Law?
overlap = filtered[filtered["case_type"] == "Family Law"]
print("Keyword matches that are also Family Law:", len(overlap))

# how many keyword matches are from OTHER case types (e.g. Civil Law mentioning cruelty)
print("Keyword matches from other case types:")
print(filtered["case_type"].value_counts())

# the cleanest, most defensible subset: Family Law + divorce keywords
final_dataset = filtered[filtered["case_type"] == "Family Law"]

print("Final dataset size:", len(final_dataset))
print("\nSample case titles:")
print(final_dataset["case_title"].head(10).to_string(index=False))

final_dataset.to_csv("final_divorce_cases.csv", index=False)
print("\nSaved to final_divorce_cases.csv")

# print ALL 84 titles so you can manually review them
pd.set_option('display.max_colwidth', None)
for i, title in enumerate(final_dataset["case_title"], 1):
    print(i, title)

# case titles confirmed as NOT divorce cases (state/criminal/service matters)
clear_removals = [
    "H D Sikand D Th Lrs vs C B I Anr on 15 December 2016",
    "Ram Saran Varshney Ors vs State Of U P Anr on 5 February 2016",
    "State Of H P And Ors vs Hem Singh on 13 February 2017",
    "Jagjit Singh vs State Of Punjab on 26 September 2018",
    "Mrs Kanika Goel vs The State Of Delhi Thru Sho on 20 July 2018",
    "Union Of India And Anr vs V R Tripathi on 11 December 2018",
    "Jagbir Singh vs State Nct Of Delhi on 4 September 2019",
    "Mahendra Prasad Mehta vs The State Of Bihar on 9 April 2019",
    "Sandeep Kumar vs The State Of Uttarakhand on 2 December 2020",
    "Central Coalfields Limited vs Parden Oraon on 9 April 2021",
    "Deepika Singh vs Central Administrative Tribunal on 16 August 2022",
    "Shafiya Khan Shakuntala Prajapati vs State Of U P on 10 February 2022",
    "Shambhu Kharwar vs The State Of Uttar Pradesh on 12 August 2022",
    "Abhishek vs The State Of Madhya Pradesh on 31 August 2023",
    "Digambar vs The State Of Maharashtra on 20 December 2024",
    "Kailashben Mahendrabhai Patel vs The State Of Maharashtra on 25 September 2024",
    "Mariam Fasihuddin vs State By Adugodi Police Station on 22 January 2024",
    "Mohd Abdul Samad vs The State Of Telangana on 10 July 2024",
    "Mulakala Malleshwara Rao vs The State Of Telangana on 29 August 2024",
    "Prakash vs The State Of Maharashtra on 20 December 2024",
    "Karan Singh vs The State Of Haryana Home Department on 31 January 2025",
    "The State Of Uttarakhand Law And Justice vs Sanjay Ram Tamta Sanju Prem Prakash on 11 February 2025",
]

# borderline: real matrimonial-adjacent cases, but likely not a divorce decree itself
borderline = [
    "Nandakumar vs The State Of Kerala on 20 April 2018",
    "Reema Salkan vs Sumer Singh Salkan on 25 September 2018",
    "Shafin Jahan vs Asokan K M on 8 March 2018",
    "Yashita Sahu vs The State Of Rajasthan on 20 January 2020",
    "Prabha Tyagi vs Kamlesh Devi on 12 May 2022",
    "Revanasiddappa vs Mallikarjun on 1 September 2023",
]

# helper: match by checking if the listed title is a substring of the actual title
# (handles the trailing "1" and any minor formatting differences in the CSV)
def title_matches_any(title, target_list):
    return any(target in str(title) for target in target_list)

clean_final = final_dataset[~final_dataset["case_title"].apply(lambda t: title_matches_any(t, clear_removals))]
borderline_cases = clean_final[clean_final["case_title"].apply(lambda t: title_matches_any(t, borderline))]
clean_final_no_borderline = clean_final[~clean_final["case_title"].apply(lambda t: title_matches_any(t, borderline))]

print("Removed as clearly not divorce:", len(final_dataset) - len(clean_final))
print("Borderline cases set aside for review:", len(borderline_cases))
print("Final clean dataset (excluding borderline):", len(clean_final_no_borderline))

clean_final_no_borderline.to_csv("clean_divorce_cases.csv", index=False)
borderline_cases.to_csv("borderline_cases_for_review.csv", index=False)

print("\nSaved clean_divorce_cases.csv and borderline_cases_for_review.csv")