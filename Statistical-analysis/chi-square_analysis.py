import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load your DataFrame
df = pd.read_excel("experiment_results.xlsx")

# Create a binary indicator column for "Brow lowering" for the relevant facial features column.
# This example checks if the text "Brow lowering" appears in the "Influential_Features-Eyebrows" column.
# You could also combine multiple columns if desired.
df["Brow_lowering_selected"] = df["Influential_Features-Eyebrows"].fillna("").apply(
    lambda x: "Brow lowering" in x
)

df['Correct_Classification'] = (
    ((df['Clip_Type'].isin(['TP', 'FN'])) & (df['Confidence_Level'].isin(['Somewhat Likely', 'Very Likely']))) |
    ((df['Clip_Type'].isin(['TN', 'FP'])) & (df['Confidence_Level'].isin(['Somewhat Unlikely', 'Very Unlikely'])))
)

# Now create a cross-tabulation with the Correct_Classification column (which you already defined)
ct_feature_correct = pd.crosstab(df["Brow_lowering_selected"], df["Correct_Classification"])
print("\nBrow Lowering vs. Correct Classification:")
print(ct_feature_correct)

# Run the chi-square test (or Fisher's exact test if the counts are small)
chi2_feature, p_feature, dof_feature, expected_feature = stats.chi2_contingency(ct_feature_correct)
print("\nChi-square test for Brow Lowering vs. Correct Classification:")
print(f"Chi2 = {chi2_feature:.3f}, p-value = {p_feature:.3f}, dof = {dof_feature}")

# If some cells have very low counts, consider using Fisher's exact test (for 2x2 tables):
if ct_feature_correct.shape == (2, 2):
    oddsratio, p_fisher = stats.fisher_exact(ct_feature_correct)
    print(f"Fisher's exact test p-value: {p_fisher:.3f}")


# -------------------- Create Combined Facial Features Column -------------------- #
def parse_features(feature_str):
    if pd.isnull(feature_str) or not isinstance(feature_str, str):
        return []
    return [f.strip() for f in feature_str.split(";") if f.strip()]

def get_all_features(row):
    eyebrows = parse_features(row["Influential_Features-Eyebrows"])
    eyes = parse_features(row["Influential_Features-Eyes"])
    mouth = parse_features(row["Influential_Features-Mouth"])
    return set(eyebrows + eyes + mouth)

df["All_Facial_Features"] = df.apply(get_all_features, axis=1)

# -------------------- Get List of All Unique Features -------------------- #
all_features = set()
for features_set in df["All_Facial_Features"]:
    all_features.update(features_set)
all_features = sorted(all_features)

# -------------------- Run Chi-Square Tests for Each Feature -------------------- #
results = []  # list to store (feature, chi2, p-value, dof, counts)
for feat in all_features:
    # Create a binary indicator: 1 if the feature is present in that row, else 0
    df[f"{feat}_present"] = df["All_Facial_Features"].apply(lambda x: 1 if feat in x else 0)
    
    # Create contingency table: rows = presence (0/1), columns = Correct_Classification (False/True)
    ct = pd.crosstab(df[f"{feat}_present"], df["Correct_Classification"])
    
    # Check if table is 2x2; if some cells have very low counts, consider Fisher's exact test.
    if ct.shape == (2, 2):
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        # Alternatively, if counts are very low, use Fisher's test:
        # oddsratio, p = stats.fisher_exact(ct)
    else:
        # If only one row exists (feature never selected), skip
        continue
    
    results.append((feat, chi2, p, dof, ct.copy()))

# Convert results to a DataFrame
results_df = pd.DataFrame(results, columns=["Feature", "Chi2", "p-value", "dof", "Contingency_Table"])

# Filter to only those with p < 0.05 (significant)
significant_features = results_df[results_df["p-value"] < 0.05]

print("=== Significant Associations: Facial Feature Presence vs. Correct Classification ===")
if significant_features.empty:
    print("No facial features show a statistically significant association with Correct Classification (p < 0.05).")
else:
    # For readability, drop the contingency tables from the printed summary
    print(significant_features.drop(columns=["Contingency_Table"]).to_string(index=False, float_format="%.3f"))

# Optionally, you could sort by p-value:
significant_features_sorted = significant_features.sort_values("p-value")
print("\nSorted Significant Features (lowest p-values first):")
print(significant_features_sorted.drop(columns=["Contingency_Table"]).to_string(index=False, float_format="%.3f"))


# -------------------- Vocal features -------------------- #
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load your DataFrame
df = pd.read_excel("experiment_results.xlsx")

# Make sure Correct_Classification is already defined (as in your previous code)
df['Correct_Classification'] = (
    ((df['Clip_Type'].isin(['TP', 'FN'])) & (df['Confidence_Level'].isin(['Somewhat Likely', 'Very Likely']))) |
    ((df['Clip_Type'].isin(['TN', 'FP'])) & (df['Confidence_Level'].isin(['Somewhat Unlikely', 'Very Unlikely'])))
)

# -------------------- Voice Features Analysis -------------------- #

# A helper to parse semicolon-separated voice features
def parse_voice_features(feature_str):
    if pd.isnull(feature_str) or not isinstance(feature_str, str):
        return []
    return [f.strip() for f in feature_str.split(";") if f.strip()]

# Create a column that contains the list of voice features per response
df["All_Voice_Features"] = df["Influential_Features-Voice"].apply(parse_voice_features)

# 1) Gather all unique voice features
all_voice_features = set()
for features in df["All_Voice_Features"]:
    all_voice_features.update(features)
all_voice_features = sorted(all_voice_features)

# 2) Run Chi-square tests for each voice feature
voice_results = []  # list to store (feature, chi2, p-value, dof, contingency table)
for feat in all_voice_features:
    # Create a binary indicator: 1 if the feature is present in the voice features, else 0
    df[f"{feat}_voice_present"] = df["All_Voice_Features"].apply(lambda x: 1 if feat in x else 0)
    
    # Create contingency table: rows = presence (0/1), columns = Correct_Classification (False/True)
    ct_voice = pd.crosstab(df[f"{feat}_voice_present"], df["Correct_Classification"])
    if ct_voice.shape == (2, 2):
        chi2_voice, p_voice, dof_voice, expected_voice = stats.chi2_contingency(ct_voice)
    else:
        # If the feature was never selected (or always selected), skip it.
        continue
    
    voice_results.append((feat, chi2_voice, p_voice, dof_voice, ct_voice.copy()))

# Convert results to a DataFrame
voice_results_df = pd.DataFrame(voice_results, columns=["Feature", "Chi2", "p-value", "dof", "Contingency_Table"])

# Filter to only those with p < 0.05 (significant associations)
significant_voice_features = voice_results_df[voice_results_df["p-value"] < 0.05]

print("=== Significant Associations: Voice Feature Presence vs. Correct Classification ===")
if significant_voice_features.empty:
    print("No voice features show a statistically significant association with Correct Classification (p < 0.05).")
else:
    # For readability, drop the contingency tables from the printed summary
    print(significant_voice_features.drop(columns=["Contingency_Table"]).to_string(index=False, float_format="%.3f"))

# Optionally, sort by p-value:
significant_voice_features_sorted = significant_voice_features.sort_values("p-value")
print("\nSorted Significant Voice Features (lowest p-values first):")
print(significant_voice_features_sorted.drop(columns=["Contingency_Table"]).to_string(index=False, float_format="%.3f"))
    