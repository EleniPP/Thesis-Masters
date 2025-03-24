import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
# Read each sheet into its own DataFrame
df1 = pd.read_excel("experiment_results.xlsx", sheet_name="table")
df2 = pd.read_excel("experiment_results.xlsx", sheet_name="salient-segments")

# Concatenate the two DataFrames (stack them)
df = pd.concat([df1, df2], ignore_index=True)

# Load your DataFrame
# df = pd.read_excel("experiment_results.xlsx")

# A helper to parse semicolon-separated features
def parse_features(feature_str):
    if pd.isnull(feature_str) or not isinstance(feature_str, str):
        return []
    return [f.strip() for f in feature_str.split(";") if f.strip()]

# Create a function to flatten eyebrows/eyes/mouth columns into one set of features
def get_all_features(row):
    # Parse each column
    eyebrows = parse_features(row["Influential_Features-Eyebrows"])
    eyes = parse_features(row["Influential_Features-Eyes"])
    mouth = parse_features(row["Influential_Features-Mouth"])
    # Combine them into a single set (so duplicates are removed)
    combined = set(eyebrows + eyes + mouth)
    return combined

# Create a function to extract voice features from each row
def get_voice_features(row):
    # Parse the "Influential_Features-Voice" column; this may contain multiple entries
    voice_features = parse_features(row["Influential_Features-Voice"])
    # Return as a set (to remove duplicates)
    return set(voice_features)

def top_cooccurrences(coocc_df, all_features, top_n=5):
    """
    Return a sorted list of the top co-occurring feature pairs in coocc_df.
    coocc_df is a symmetric matrix of shape (M, M) with co-occurrence counts.
    all_features is a list of feature names in the same order as coocc_df rows/cols.
    top_n is how many top pairs to return.
    """
    coocc_list = []
    M = len(all_features)
    
    # Only look at the upper triangle (i < j) to avoid duplicating pairs
    for i in range(M):
        for j in range(i+1, M):
            count = coocc_df.iloc[i, j]
            coocc_list.append((all_features[i], all_features[j], count))
    
    # Sort by count descending
    coocc_list.sort(key=lambda x: x[2], reverse=True)
    
    return coocc_list[:top_n]

# Facial features

# # Apply this to each row
# df["All_Facial_Features"] = df.apply(get_all_features, axis=1)

# # 1) Gather all unique features
# all_features = set()
# for features_set in df["All_Facial_Features"]:
#     all_features.update(features_set)

# all_features = sorted(all_features)  # sorted list for consistent ordering

# # 2) Initialize a 2D matrix of zeros
# feature_to_idx = {feat: i for i, feat in enumerate(all_features)}
# n = len(all_features)
# coocc_matrix = np.zeros((n, n), dtype=int)
# feature_count = np.zeros(n, dtype=int)

# # 3) Fill the co-occurrence counts
# for features_set in df["All_Facial_Features"]:
#     # Convert the set to a list to iterate pairs
#     feats_list = list(features_set)
#     # increment the single-feature count for each feature
#     for f in feats_list:
#         i = feature_to_idx[f]
#         feature_count[i] += 1
#     # For each pair in this set, increment co-occurrence
#     for f1, f2 in itertools.combinations(feats_list, 2):
#         i1 = feature_to_idx[f1]
#         i2 = feature_to_idx[f2]
#         coocc_matrix[i1, i2] += 1
#         coocc_matrix[i2, i1] += 1

# # coocc_matrix[i, j] now tells you how many times feature i co-occurs with feature j
# coocc_df = pd.DataFrame(coocc_matrix, index=all_features, columns=all_features)
# # print("=== Co-Occurrence Matrix (Raw Counts) ===")
# # print(coocc_df)

# # Option A: fraction of total responses
# coocc_fraction = coocc_df / len(df) * 100

# # Option B: fraction of times feature i is chosen
# # For each feature i, find how many times it appears at all, then scale
# feature_counts = coocc_df.values.diagonal()  # co-occ_df[i,i] is how often feature i is chosen
# coocc_percentage = pd.DataFrame(index=all_features, columns=all_features, dtype=float)

# for i, fi in enumerate(all_features):
#     denom = feature_counts[i]
#     if denom == 0:
#         coocc_percentage.loc[fi, :] = 0
#     else:
#         coocc_percentage.loc[fi, :] = (coocc_df.loc[fi, :] / denom) * 100

# plt.figure(figsize=(14,10))
# sns.heatmap(coocc_df, annot=True, fmt="d", cmap="Reds",
#             xticklabels=all_features, yticklabels=all_features)
# plt.title("Co-Occurrence of Facial Features (Raw Counts)")
# # Rotate the x-axis tick labels by 45 degrees, align them to the right
# plt.xticks(rotation=45, ha='right')
# # Add extra padding at the bottom to accommodate labels
# plt.subplots_adjust(bottom=0.25, left = 0.25)
# plt.show()


# # Example usage:
# top_pairs = top_cooccurrences(coocc_df, all_features, top_n=5)

# # print("Top 5 Co-Occurring Feature Pairs:")
# # for f1, f2, count in top_pairs:
# #     print(f"{f1} & {f2} => {count} times")


# # Chi-Square Test on All Pairs
# # ======================
# # For each pair, build a 2x2 contingency table and compute the chi-square test.
# # Chi-Square Test on All Pairs (with zero-frequency check)
# chi_square_results = []
# N = len(df)  # total number of obserevations
# M = len(all_features)
# for i in range(M):
#     for j in range(i+1, M):
#         AB = coocc_matrix[i, j]  # both features present
#         A = feature_count[i]   # total count for feature i
#         B = feature_count[j]   # total count for feature j

#         # Skip if either feature never appears (to avoid zero expected frequencies)
#         if A == 0 or B == 0:
#             continue

#         # Compute remaining counts for the 2x2 contingency table
#         neither = N - A - B + AB
#         onlyA = A - AB
#         onlyB = B - AB
        
#         contingency = [[AB, onlyA],
#                        [onlyB, neither]]
        
#         chi2, p_value, dof, expected = chi2_contingency(contingency)

#         print("Contingency Table:")
#         print(contingency)
#         print("Expected Frequencies:")
#         print(expected)

#         if (expected < 5).any():
#             print("Warning: Some expected frequencies are below 5. Chi-square results may not be reliable.")
#         else:
#             print("All expected frequencies are 5 or above.")

#         chi_square_results.append((all_features[i], all_features[j], AB, p_value, chi2))

# # Filter for significant pairs (p < 0.05)
# significant_pairs = [res for res in chi_square_results if res[3] < 0.05]
# # Sort significant pairs by descending co-occurrence count (or by p-value)
# significant_pairs.sort(key=lambda x: x[2], reverse=True)

# print("\nSignificant Co-Occurring Feature Pairs (Chi-Square, p < 0.05):")
# for f1, f2, count, p_value, chi2 in significant_pairs:
#     print(f"{f1} & {f2}: Co-occurrence={count}, chi2={chi2:.2f}, p-value={p_value:.4f}")



# # Convert to a DataFrame
# df_top_pairs = pd.DataFrame(top_pairs, columns=["Feature1", "Feature2", "Count"])

# # Create a single label for each pair, e.g., "Feature1 & Feature2"
# df_top_pairs["Pair"] = df_top_pairs["Feature1"] + " & " + df_top_pairs["Feature2"]

# plt.figure(figsize=(8, 6))
# ax = sns.barplot(data=df_top_pairs, x="Count", y="Pair", color="skyblue")
# plt.title("Top 10 Co-occurring Voice Feature Pairs")
# plt.xlabel("Co-occurrence Count")
# plt.ylabel("Feature Pair")
# # Adjust the bar thickness (height) for each horizontal bar
# for bar in ax.patches:
#     bar.set_height(0.3)  # Adjust this value as needed
# plt.tight_layout()
# plt.show()


# ----------------- Voice Features -----------------
# Apply the function to create a new column with all voice features per response
df["All_Voice_Features"] = df.apply(get_voice_features, axis=1)

# 1) Gather all unique voice features from the DataFrame
all_voice_features = set()
for feature_set in df["All_Voice_Features"]:
    all_voice_features.update(feature_set)
all_voice_features = sorted(all_voice_features)  # sorted for consistency

# 2) Initialize a 2D co-occurrence matrix (size = number of unique voice features)
voice_to_idx = {feat: i for i, feat in enumerate(all_voice_features)}
n_voice = len(all_voice_features)
coocc_matrix_voice = np.zeros((n_voice, n_voice), dtype=int)
feature_count = np.zeros(n_voice, dtype=int)

# 3) Fill the co-occurrence counts for each response
for feature_set in df["All_Voice_Features"]:
    # Convert the set to a list
    feats_list = list(feature_set)
    for f in feats_list:
        i = voice_to_idx[f]
        feature_count[i] += 1
    # For each unique pair in the response, increment the counts
    for f1, f2 in itertools.combinations(feats_list, 2):
        i1 = voice_to_idx[f1]
        i2 = voice_to_idx[f2]
        coocc_matrix_voice[i1, i2] += 1
        coocc_matrix_voice[i2, i1] += 1

# Wrap the matrix in a pandas DataFrame for readability
coocc_df_voice = pd.DataFrame(coocc_matrix_voice, index=all_voice_features, columns=all_voice_features)
print("=== Co-Occurrence Matrix for Voice Features (Raw Counts) ===")
print(coocc_df_voice)

# 4) Plot a heatmap of the co-occurrence matrix
plt.figure(figsize=(14,10))
# Change the color palette by modifying cmap: options include "coolwarm", "viridis", "inferno", etc.
sns.heatmap(coocc_df_voice, annot=True, fmt="d", cmap="Blues",
            xticklabels=all_voice_features, yticklabels=all_voice_features)
plt.title("Co-Occurrence of Voice Features (Raw Counts)")
plt.xticks(rotation=35, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.subplots_adjust(left=0.3)
plt.show()

def top_cooccurrences(coocc_df, features, top_n=5):
    """
    Return a sorted list of the top co-occurring feature pairs in coocc_df.
    coocc_df: pandas DataFrame of co-occurrence counts (symmetric matrix)
    features: list of feature names corresponding to the rows/columns of coocc_df
    top_n: number of top pairs to return
    """
    coocc_list = []
    M = len(features)
    # Only iterate over the upper triangle (i < j)
    for i in range(M):
        for j in range(i+1, M):
            count = coocc_df.iloc[i, j]
            coocc_list.append((features[i], features[j], count))
    # Sort by count in descending order
    coocc_list.sort(key=lambda x: x[2], reverse=True)
    return coocc_list[:top_n]

# Calculate the top 5 co-occurring voice feature pairs
top_voice_pairs = top_cooccurrences(coocc_df_voice, all_voice_features, top_n=5)

print("Top 5 Co-Occurring Voice Feature Pairs:")
for f1, f2, count in top_voice_pairs:
    print(f"{f1} & {f2} => {count} times")

chi_square_results = []
N = len(df)  # total number of obserevations
M = len(all_voice_features)
for i in range(M):
    for j in range(i+1, M):
        AB = coocc_matrix_voice[i, j]  # both features present
        A = feature_count[i]   # total count for feature i
        B = feature_count[j]   # total count for feature j

        # Skip if either feature never appears (to avoid zero expected frequencies)
        if A == 0 or B == 0:
            continue

        # Compute remaining counts for the 2x2 contingency table
        neither = N - A - B + AB
        onlyA = A - AB
        onlyB = B - AB
        
        contingency = [[AB, onlyA],
                       [onlyB, neither]]
        
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        print("Contingency Table:")
        print(contingency)
        print("Expected Frequencies:")
        print(expected)

        if (expected < 5).any():
            print("Warning: Some expected frequencies are below 5. Chi-square results may not be reliable.")
        else:
            print("All expected frequencies are 5 or above.")

        chi_square_results.append((all_voice_features[i], all_voice_features[j], AB, p_value, chi2))

# Filter for significant pairs (p < 0.05)
significant_pairs = [res for res in chi_square_results if res[3] < 0.05]
# Sort significant pairs by descending co-occurrence count (or by p-value)
significant_pairs.sort(key=lambda x: x[2], reverse=True)

print("\nSignificant Co-Occurring Feature Pairs (Chi-Square, p < 0.05):")
for f1, f2, count, p_value, chi2 in significant_pairs:
    print(f"{f1} & {f2}: Co-occurrence={count}, chi2={chi2:.2f}, p-value={p_value:.4f}")
