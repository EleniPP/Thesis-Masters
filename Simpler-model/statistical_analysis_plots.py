import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pingouin as pg
import scipy.stats as stats
import numpy as np

def parse_interval(interval_str):
    """
    Given a string like '2-3 sec' or '4.4-5.9 sec',
    return a tuple (start_float, end_float).
    """
    # Remove the ' sec' part (if it exists)
    interval_str = interval_str.replace(" sec", "").strip()
    
    # Split by the dash
    start_str, end_str = interval_str.split("-")
    
    # Convert to float
    start_val = float(start_str)
    end_val = float(end_str)
    
    return start_val, end_val

def within_margin(sel_start, sel_end, model_start, model_end, margin=1.0):
    start_close = abs(sel_start - model_start) <= margin
    end_close = abs(sel_end - model_end) <= margin
    return start_close and end_close

folder = os.path.join(os.path.expanduser("~"), "Downloads")
# Load sample data (replace with actual data structure)
# data = pd.read_csv(folder + "/experiment_results.csv")  # Assuming structured CSV with participant responses
# print(data)
# # Convert categorical variables into numerical values if needed
# confidence_mapping = {'Very Unlikely': 1, 'Somewhat Unlikely': 2, 'Somewhat Likely': 3, 'Very Likely': 4}
# data['Confidence'] = data['Confidence_Level'].map(confidence_mapping)

# print(data)
# print(1/0)

# Load data from excel sheet table
data = pd.read_excel('experiment_results.xlsx', sheet_name='table')
confidence_mapping = {'Very Unlikely': 1, 'Somewhat Unlikely': 2, 'Somewhat Likely': 3, 'Very Likely': 4}
data['Confidence'] = data['Confidence_Level'].map(confidence_mapping)

data[["Selected_Start", "Selected_End"]] = data["Selected_Time_Interval"].apply(
    lambda x: pd.Series(parse_interval(x))
)

data[["Model_Start", "Model_End"]] = data["Model_Salient_Interval"].apply(
    lambda x: pd.Series(parse_interval(x))
)

data["Within_1s_Margin"] = data.apply(
    lambda row: within_margin(
        row["Selected_Start"], row["Selected_End"],
        row["Model_Start"],    row["Model_End"],
        margin=1.0
    ),
    axis=1
)

# ----------------- 1. Comparison of Human-Selected Salient Intervals vs. Model-Salient Segments -----------------
# A bit useless
# sns.histplot(data, x='Selected_Time_Interval', hue='Model_Salient_Interval', multiple='stack', palette='coolwarm')
# plt.title('Human vs. Model Salient Segment Selection')
# plt.xlabel('Selected Time Interval (Seconds)')
# plt.ylabel('Count')
# plt.show()

# ----------------- 2. Analysis of Influential Features by Clip Type (TP, TN, FP, FN) -----------------
# Good graph but maybe not with TP TN FP FN because it is a bit useless if they havent found the salient segment. Maybe we should do it with
# wether its classifies correctly or not


# # Melt the DataFrame to gather all influential features into one column
# data_filtered = data[data["Within_1s_Margin"] == True]
# print(data_filtered)
# data_melted = data_filtered.melt(id_vars=['Clip_Type'], 
#                          value_vars=['Influential_Features-Eyebrows', 
#                                      'Influential_Features-Eyes', 
#                                      'Influential_Features-Mouth'],
#                          var_name='Feature_Type', 
#                          value_name='Influential_Feature')

# # Ignore empty or placeholder values ('-' or NaN) but keep the row
# data_melted = data_melted[data_melted['Influential_Feature'].notna()]
# data_melted = data_melted[data_melted['Influential_Feature'] != '-']

# # Rename 'Feature_Type' values for better readability
# data_melted['Feature_Type'] = data_melted['Feature_Type'].str.replace('Influential_Features-', '')

# # **NEW STEP: Split multi-feature selections into separate rows**
# data_melted = data_melted.assign(Influential_Feature=data_melted['Influential_Feature'].str.split('; '))
# data_melted = data_melted.explode('Influential_Feature')

# # Split into three DataFrames
# df_eyebrows = data_melted[data_melted['Feature_Type'] == 'Eyebrows']
# df_eyes = data_melted[data_melted['Feature_Type'] == 'Eyes']
# df_mouth = data_melted[data_melted['Feature_Type'] == 'Mouth']

# # Set up subplots
# fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# # Define colors
# colors = {'Eyebrows': 'purple', 'Eyes': 'blue', 'Mouth': 'orange'}

# # Plot for Eyebrows
# sns.countplot(data=df_eyebrows, x='Clip_Type', hue='Influential_Feature', palette='Purples', ax=axes[0])
# axes[0].set_title('Influential Eyebrow Features by Clip Type')
# axes[0].set_ylabel('Feature Count')
# axes[0].legend(title='Eyebrow Features', bbox_to_anchor=(1, 1))

# # Plot for Eyes
# sns.countplot(data=df_eyes, x='Clip_Type', hue='Influential_Feature', palette='Blues', ax=axes[1])
# axes[1].set_title('Influential Eye Features by Clip Type')
# axes[1].set_ylabel('Feature Count')
# axes[1].legend(title='Eye Features', bbox_to_anchor=(1, 1))

# # Plot for Mouth
# sns.countplot(data=df_mouth, x='Clip_Type', hue='Influential_Feature', palette='Oranges', ax=axes[2])
# axes[2].set_title('Influential Mouth Features by Clip Type')
# axes[2].set_xlabel('Clip Classification')
# axes[2].set_ylabel('Feature Count')
# axes[2].legend(title='Mouth Features', bbox_to_anchor=(1, 1))

# # Show the plots
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# ----------------- 3. Confidence Level Distribution per Clip Type -----------------
# The confidence level depending on the category (TP,TN,FP,FN)
# plt.figure(figsize=(8, 5))
# sns.boxplot(x='Clip_Type', y='Confidence', data=data, palette='Set2')
# plt.title('Confidence Level by Clip Classification')
# plt.xlabel('Clip Classification')
# plt.ylabel('Confidence Score')
# plt.show()

# # The confidence level depending on the Misclassification Type (Correct, Shared Mistake, Divergent Mistake)
# # Reverse confidence mapping for better readability
# confidence_map = {
#     1: 'Very Unlikely',
#     2: 'Somewhat Unlikely',
#     3: 'Somewhat Likely',
#     4: 'Very Likely'
# }

# plt.figure(figsize=(10, 6))  # Larger figure for better visibility
# ax = sns.boxplot(
#     x='Misclassification_Type', 
#     y='Confidence', 
#     data=data, 
#     width=0.3,
#     palette='Set2'
# )

# plt.title('Confidence Level by Misclassification Type', fontsize=14)
# plt.xlabel('Misclassification Type', fontsize=12)
# plt.ylabel('Confidence Level', fontsize=12)

# # Set custom ticks & labels
# ax.set_yticks([1,2,3,4])
# ax.set_yticklabels([confidence_map[i] for i in range(1, 5)], fontsize=12)

# # Optional: Give a little vertical padding so nothing gets cut off
# plt.ylim(0.8, 4.3)

# # Ensure everything fits in the figure
# plt.tight_layout()
# plt.show()

# ----------------- 4. Time Interval Selection Distribution per Clip Type -----------------
# I dont think that gives any useful information
# plt.figure(figsize=(10, 5))
# sns.histplot(data, x='Selected_Time_Interval', hue='Clip_Type', multiple='stack', palette='husl')
# plt.title('Time Interval Selection by Clip Type')
# plt.xlabel('Selected Time Interval (Seconds)')
# plt.ylabel('Count')
# plt.show()

# ----------------- 6. Misclassification & Confidence Analysis -----------------
# confidence_map = {
#     1: 'Very Unlikely',
#     2: 'Somewhat Unlikely',
#     3: 'Somewhat Likely',
#     4: 'Very Likely'
# }

# plt.figure(figsize=(10, 5))
# sns.boxplot(x='Misclassification_Type', y='Confidence', data=data, width=0.3,palette='muted')
# plt.title('Confidence Level in Misclassified Cases')
# plt.xlabel('Misclassification Type (Shared/Divergent)')
# plt.ylabel('Confidence Score')
# plt.yticks([1, 2, 3, 4],
#            [confidence_map[1], confidence_map[2], confidence_map[3], confidence_map[4]])
# plt.tight_layout()
# plt.show()


# ----------------- Miss Rate by Salient Segment Position -----------------
# Group data by Salient Segment Position Type and calculate miss rate
# miss_rate_by_position = data.groupby('Salient_Position_Type')['Within_1s_Margin'].apply(lambda x: (x == False).mean()).reset_index()

# # Rename column for clarity
# miss_rate_by_position.rename(columns={'Within_1s_Margin': 'Miss_Rate'}, inplace=True)

# # Plot bar chart
# plt.figure(figsize=(8, 5))
# sns.barplot(data=miss_rate_by_position, x='Salient_Position_Type', y='Miss_Rate', palette='coolwarm')
# plt.title('Miss Rate by Salient Segment Position')
# plt.xlabel('Salient Segment Position')
# plt.ylabel('Percentage of Participants Missing Salient Segment')
# plt.ylim(0, 1)  # Normalize y-axis (0 = no misses, 1 = all participants missed)
# plt.xticks(rotation=45)
# plt.show()

# ----------------- Classification Rate by Salient Segment Position -----------------
# Define correct classification based on salient segment type and participant response
data['Correct_Classification'] = (
    ((data['Clip_Type'].isin(['TP', 'FN'])) & (data['Confidence_Level'].isin(['Somewhat Likely', 'Very Likely']))) |
    ((data['Clip_Type'].isin(['TN', 'FP'])) & (data['Confidence_Level'].isin(['Somewhat Unlikely', 'Very Unlikely'])))
)

# Compute correct classification rate per salient segment position
classification_accuracy = data.groupby('Salient_Position_Type')['Correct_Classification'].mean().reset_index()

# Rename column for clarity
classification_accuracy.rename(columns={'Correct_Classification': 'Classification_Accuracy'}, inplace=True)

# # Plot bar chart
# plt.figure(figsize=(8, 5))
# sns.barplot(data=classification_accuracy, x='Salient_Position_Type', y='Classification_Accuracy', palette='coolwarm')
# plt.title('Correct Classification Rate by Salient Segment Position')
# plt.xlabel('Salient Segment Position')
# plt.ylabel('Percentage of Participants Classifying Clip Correctly')
# plt.ylim(0, 1)  # Normalize y-axis (0 = 0%, 1 = 100%)
# plt.xticks(rotation=45)
# plt.show()

# ----------------- Percentage of Participants Within 1s by Classification Accuracy -----------------
# # Compute proportions of participants within and outside 1s for correct and incorrect classifications
# within_1s_distribution = data.groupby('Correct_Classification')['Within_1s_Margin'].value_counts(normalize=True).unstack().reset_index()

# # Rename columns
# within_1s_distribution.rename(columns={True: 'Within_1s', False: 'Outside_1s'}, inplace=True)

# # Convert to percentage
# within_1s_distribution[['Within_1s', 'Outside_1s']] *= 100

# # Plot stacked bar chart
# within_1s_distribution.set_index('Correct_Classification').plot(kind='bar', stacked=True, figsize=(8, 6), colormap='coolwarm')
# plt.title('Within 1s vs. Outside 1s by Classification Accuracy')
# plt.xlabel('Clip Classified Correctly?')
# plt.ylabel('Percentage')
# plt.legend(title='Within 1s of Model-Salient Segment', labels=['Outside 1s', 'Within 1s'])
# plt.xticks(ticks=[0, 1], labels=['Incorrect', 'Correct'])
# plt.xticks(rotation=45, ha='right')  # Rotate labels 45 degrees and align them to the right
# plt.show()


# ----------------- Response Bias Analysis -----------------
# plt.figure(figsize=(8, 5))
# sns.countplot(x='Selected_Time_Interval', data=data, palette='coolwarm')
# plt.title('Distribution of Selected Time Intervals')
# plt.xlabel('Time Interval')
# plt.ylabel('Count of Selections')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Count the frequency of each interval
# observed = data['Selected_Time_Interval'].value_counts().sort_index()
# print("Observed counts:\n", observed)

# # Expected count: if responses were uniformly distributed,
# # each interval would be selected equally often.
# n_intervals = len(observed)
# expected_count = observed.sum() / n_intervals
# expected = [expected_count] * n_intervals

# # Perform the Chi-square goodness-of-fit test
# chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
# print("Chi-square statistic: {:.2f}, p-value: {:.4f}".format(chi2, p))

# # 'Participant_ID', 'Clip_ID', and 'Selected_Time_Interval'
# # If Selected_Time_Interval is categorical (e.g., "0-1 sec"), consider encoding it as an ordinal variable
# # For example, create a new column that maps each interval to a numeric code:
# interval_order = {"0-1 sec": 1, "1-2 sec": 2, "2-3 sec": 3, "3-4 sec": 4, "4-5 sec": 5,
#                   "5-6 sec": 6, "6-7 sec": 7, "7-8.5 sec": 8}
# data['Interval_Code'] = data['Selected_Time_Interval'].map(interval_order)

# # Now run repeated measures ANOVA where each participant is measured across different Clip_IDs
# rm_results = pg.rm_anova(dv='Interval_Code', within='Clip_ID', subject='Participant_ID', data=data, detailed=True)
# print(rm_results)

# ----------------------------------------------------------------------------------------------------------------
# xl = pd.ExcelFile('experiment_results.xlsx')
# data_unit2 = pd.read_excel('experiment_results.xlsx', sheet_name='salient-segments')

# print(data.shape)
# print(data.head())
# print(data.columns)

# print(data_unit2.shape)
# print(data_unit2.head())
# print(data_unit2.columns)

# merged = pd.merge(data, data_unit2, on=["Participant_ID", "Clip_ID"], suffixes=('_participant', '_model'))

# def to_feature_set(features_str):
#     """Convert a semicolon-separated string to a set of stripped features."""
#     if pd.isna(features_str):
#         return set()
#     return set(f.strip() for f in features_str.split(';') if f.strip())

# # For each category, compute overlap
# for category in ['Influential_Features-Eyebrows', 'Influential_Features-Eyes',
#                  'Influential_Features-Mouth', 'Influential_Features-Voice']:
#     participant_col = f"{category}_participant"
#     model_col = f"{category}_model"
#     if participant_col in merged.columns and model_col in merged.columns:
#         # Convert strings to sets
#         merged[f"{category}_participant_set"] = merged[participant_col].apply(to_feature_set)
#         merged[f"{category}_model_set"] = merged[model_col].apply(to_feature_set)

#         # Calculate intersection, union, or other overlap measures
#         merged[f"{category}_overlap"] = merged.apply(
#             lambda row: row[f"{category}_participant_set"].intersection(
#                 row[f"{category}_model_set"]), axis=1
#         )
#         merged[f"{category}_overlap_count"] = merged[f"{category}_overlap"].apply(len)

# # Inspect results
# print(merged)

# ----------------------------------------------------------------------------------------------------------------
df = pd.read_excel('experiment_results.xlsx', sheet_name='table')

def split_features(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in s.split(';') if x.strip()]

# Process each of the three feature columns separately
feature_columns = ['Influential_Features-Eyebrows', 
                   'Influential_Features-Eyes', 
                   'Influential_Features-Mouth']

for col in feature_columns:
    df[col + '_list'] = df[col].apply(split_features)

# Combine the lists from the three columns into one column per row
df['All_Features'] = df[[col + '_list' for col in feature_columns]].apply(
    lambda row: row[0] + row[1] + row[2], axis=1
)
# Group by Clip_ID (assuming there's a Clip_ID column)
grouped = df.groupby('Clip_ID')

# Create a dictionary to hold the radar data for each clip
radar_data_by_clip = {}

for clip_id, group in grouped:
    # Explode the All_Features column for the current clip
    exploded = group.explode('All_Features')
    feature_counts = exploded['All_Features'].value_counts().reset_index()
    feature_counts.columns = ['axis', 'value']

        # Print the non-normalized counts for this clip
    print(f"Non-normalized counts for {clip_id}:")
    print(feature_counts)
    
    # Optional: Normalize the counts (e.g., max becomes 1)
    max_value = feature_counts['value'].max()
    feature_counts['value'] = feature_counts['value'] / max_value
    
    # Save as a list of dicts for this clip
    radar_data_by_clip[clip_id] = feature_counts.to_dict(orient='records')

# Example output for one clip:
print(radar_data_by_clip['Clip 1'])
# print(radar_data_by_clip['Clip2'])