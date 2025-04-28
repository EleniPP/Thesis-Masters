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

# Load data from excel sheet table
data = pd.read_excel('real_experiment_results.xlsx', sheet_name='table')
# confidence_mapping = {'Very Unlikely': 1, 'Somewhat Unlikely': 2, 'Somewhat Likely': 3, 'Very Likely': 4}
# data['Confidence'] = data['Confidence_Level'].map(confidence_mapping)

data["Selected_Start"] = data["Selected_Timestamp"]
data["Selected_End"] = data["Selected_Timestamp"]

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

data['Correct_Classification'] = (
    ((data['Clip_Type'].isin(['TP', 'FN'])) & (data['Confidence_Level'] > 5)) |
    ((data['Clip_Type'].isin(['TN', 'FP'])) & (data['Confidence_Level'] <= 5))
)

def get_classification_type(row):
    # Participant said "Depressed" if Confidence_Level > 5, otherwise "Non-Depressed"
    if row['Confidence_Level'] > 5:
        # Participant said depressed: if the clip is actually depressed, label it TP; otherwise, FP.
        if row['Clip_Type'] in ['TP', 'FN']:
            return 'TP'
        else:
            return 'FP'
    else:
        # Participant said non-depressed: if the clip is actually non-depressed, label it TN; otherwise, FN.
        if row['Clip_Type'] in ['TN', 'FP']:
            return 'TN'
        else:
            return 'FN'

data['Classification_Type'] = data.apply(get_classification_type, axis=1)

# Create a new column with more descriptive labels
data['Classification_Label'] = data['Correct_Classification'].map({True: 'Correct', False: 'Incorrect'})



data["Model_Mid"] = (data["Model_Start"] + data["Model_End"]) / 2


data["Selected_Timestamp"] = data["Selected_Timestamp"].astype(float)

# 2) Compute the absolute error in seconds
data["Error"] = (data["Selected_Timestamp"] - data["Model_Mid"]).abs()

# 3) Group by Clip_ID (or however you identify each clip)
error_stats = data.groupby("Clip_ID").agg(
    mean_error = ("Error", "mean"),
    std_error = ("Error", "std"),
    within_1s_pct = ("Within_1s_Margin", lambda x: 100 * x.mean())
).reset_index()

# # 4) Print or display the table
# print("=== Error Analysis by Clip ===")
# print(error_stats.to_string(index=False, float_format="%.2f"))

# Optional: rename columns for clarity or format the output
# e.g. rename "within_1s_pct" to "% Within 1s"
error_stats.rename(columns={"within_1s_pct": "%_Within_1s"}, inplace=True)

print("\n=== Error Analysis by Clip (Final) ===")
print(error_stats.to_string(index=False, float_format="%.2f"))

# ----------------- Above table but not per clip. Global analysis -----------------
# Compute overall summary statistics:
global_mean_error = data["Error"].mean()
global_std_error = data["Error"].std()
global_median_error = data["Error"].median()
global_pct_within1 = 100 * data["Within_1s_Margin"].mean()

print("Global Average Error (s):", global_mean_error)
print("Global Error STD (s):", global_std_error)
print("Global Median Error (s):", global_median_error)
print("Global % Within 1s:", global_pct_within1)

# Create a summary table (by clip or overall)
global_summary = pd.DataFrame({
    "Metric": ["Mean Error (s)", "Std Error (s)", "Median Error (s)", "% Within 1s"],
    "Global Value": [global_mean_error, global_std_error, global_median_error, global_pct_within1]
})
print(global_summary)





# ----------------- Violin plot for Raw error instead of the Box plot for MAE -----------------
data["RawError"] = data["Selected_Timestamp"] - data["Model_Mid"]

tp_tp_data = data[(data["Clip_Type"] == "TP") & (data["Classification_Type"] == "TP")]
tp_fn_data = data[(data["Clip_Type"] == "FN") & (data["Classification_Type"] == "TP")]
tn_tn_data = data[(data["Clip_Type"] == "TN") & (data["Classification_Type"] == "TN")]
tn_fp_data = data[(data["Clip_Type"] == "FP") & (data["Classification_Type"] == "TN")]

# tp_data = data[(data["Clip_Type"] == "TN") & (data["Classification_Type"] == "TN")]

# tp_data["RawError"] = tp_data["Selected_Timestamp"] - tp_data["Model_Mid"]

# Determine the global minimum and maximum of RawError across all datasets
global_min = min(df["RawError"].min() for df in [tp_tp_data, tp_fn_data, tn_tn_data, tn_fp_data])
global_max = max(df["RawError"].max() for df in [tp_tp_data, tp_fn_data, tn_tn_data, tn_fp_data])

print(global_min, global_max)

plt.figure(figsize=(8, 6))
# Plot a vertical violin plot: the y-axis is RawError.
sns.violinplot(y="RawError", data=tn_fp_data, color="skyblue")
plt.ylim(global_min-2, global_max+4)
# Add a horizontal line at y=0 (the model’s midpoint)
plt.axhline(y=0, color="red", linestyle="--", label="Model Midpoint")

plt.ylabel("Raw Error (seconds)")
plt.title("Vertical Violin Plot of Raw Error")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
tn_tn_data = data[
    (data["Clip_Type"] == "TN") & 
    (data["Classification_Type"] == "TN")
]
tp_tp_data = data[
    (data["Clip_Type"] == "TP") & 
    (data["Classification_Type"] == "TP")
]
fn_tp_data = data[
    (data["Clip_Type"] == "FN") & 
    (data["Classification_Type"] == "TP")
]
fp_tn_data = data[
    (data["Clip_Type"] == "FP") & 
    (data["Classification_Type"] == "TN")
]
# Filter data only to segments classified as TP by the model
tp_segments = data[data["Clip_Type"] == "TP"]

# Create the contingency table using Classification_Label
tp_table = pd.crosstab(tp_segments["Classification_Label"], tp_segments["Within_1s_Margin"], normalize='index')

# Rename columns clearly
tp_table.columns = ['Outside 1s', 'Within 1s']

# Convert to percentages and reorder
tp_table = tp_table[['Within 1s', 'Outside 1s']] * 100

print(tp_table.round(1))







# --------------------------------------------------------
# x = tp_data["Model_Mid"]
# y = tp_data["Selected_Timestamp"]

# # Determine min and max across both sets of times
# min_val = min(x.min(), y.min())
# max_val = max(x.max(), y.max())

# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, color='skyblue', label="Data Points")
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Alignment (y = x)")

# plt.xlabel("Model Selected Time (seconds)")
# plt.ylabel("Participant Selected Time (seconds)")
# plt.title("Scatter Plot: Model vs. Participant Timestamps")
# plt.legend()
# plt.grid(True)
# plt.show()

# # ----------------- Scatter plot with Jitter for Salient Position Type -----------------
# tp_data = data[(data["Clip_Type"] == "TP") & (data["Classification_Type"] == "TP")]

# # Get the unique cases from the Salient_Position_Type column and sort them
# unique_cases = sorted(tp_data["Salient_Position_Type"].unique())


# # Create subplots: one plot per unique case
# fig, axes = plt.subplots(nrows=1, ncols=len(unique_cases), figsize=(15, 4), sharey=True)

# for i, case_val in enumerate(unique_cases):
#     ax = axes[i]
    
#     # Subset data for the current case
#     subset = tp_data[tp_data["Salient_Position_Type"] == case_val]
    
#     # Assume there is a common model timestamp for the current case
#     model_times = subset["Model_Mid"].unique()
#     if len(model_times) == 0:
#         continue
#     model_time = model_times[0]
    
#     # Extract participant-selected timestamps
#     participant_times = subset["Selected_Timestamp"].values
    
#     # Plot a vertical line for the model's selected timestamp
#     ax.axvline(x=model_time, color='red', linestyle='--', label="Model Time")
    
#     # Plot participant timestamps as points with a small random y-jitter
#     y_jitter = np.random.uniform(-0.01, 0.01, size=len(participant_times))
#     ax.scatter(participant_times, y_jitter, color='blue', alpha=0.7, label="Participant Time")
    
#     # Enforce x-axis range from 0 to 8.5
#     ax.set_xlim(0, 8.5)
    
#     ax.set_xlabel("Time (seconds)")
#     ax.set_title(case_val)
#     ax.set_yticks([])  # Hide y-axis ticks since they are not meaningful
    
#     if i == 0:
#         ax.set_ylabel("Jitter (arbitrary)")
#     ax.legend()

# plt.suptitle("Comparison of Model and Participant Timestamps by Salient Position Type")
# plt.tight_layout()
# plt.show()



# # Plot a histogram of errors across all responses
# plt.figure(figsize=(8, 5))
# plt.hist(data["Error"], bins=20, color="skyblue", edgecolor="black")
# plt.xlabel("Absolute Error (seconds)")
# plt.ylabel("Frequency")
# plt.title("Global Distribution of Absolute Errors")
# plt.show()

# # Optionally, a boxplot for the error distribution:
# plt.figure(figsize=(4, 6))
# plt.boxplot(data["Error"], vert=True, patch_artist=True,
#             boxprops=dict(facecolor='lightgreen', color='green'),
#             medianprops=dict(color='red'))
# plt.ylabel("Absolute Error (seconds)")
# plt.title("Boxplot of Global Errors")
# plt.show()

# # Optionally, you can also compute a cumulative distribution:
# errors_sorted = np.sort(data["Error"])
# cumulative = np.arange(1, len(errors_sorted)+1) / len(errors_sorted) * 100

# plt.figure(figsize=(8, 5))
# plt.plot(errors_sorted, cumulative, marker='o', linestyle='-')
# plt.xlabel("Absolute Error (seconds)")
# plt.ylabel("Cumulative % of Responses")
# plt.title("Cumulative Distribution of Global Errors")
# plt.grid(True)
# plt.show()


# ----------------------------------------------------------------------------------------------------


# # ----------------- Percentage of Participants Within 1s by Classification Accuracy -----------------
# # Compute proportions of participants within and outside 1s for correct and incorrect classifications
# within_1s_distribution = data.groupby('Correct_Classification')['Within_1s_Margin'].value_counts(normalize=True).unstack().reset_index()

# # Rename columns
# within_1s_distribution.rename(columns={True: 'Within_1s', False: 'Outside_1s'}, inplace=True)

# # Convert to percentage
# within_1s_distribution[['Within_1s', 'Outside_1s']] *= 100

# # Plot stacked bar chart
# within_1s_distribution.set_index('Correct_Classification').plot(kind='bar', stacked=True, figsize=(8, 6), color=['#ce434a', '#48a389'])
# # plt.title('Within 1s vs. Outside 1s by Classification Accuracy')
# # plt.xlabel('Clip Classified Correctly?')
# # plt.ylabel('Percentage')
# # plt.legend(
# #     title='Within 1s of Model-Salient Segment',
# #     labels=['Outside 1s', 'Within 1s'],
# #     bbox_to_anchor=(1.05, 1),  # 5% to the right of the axes, aligned at top
# #     loc='upper left',
# #     borderaxespad=0.
# # )
# # plt.xticks(ticks=[0, 1], labels=['Incorrect', 'Correct'])
# # plt.xticks(rotation=45, ha='right')  # Rotate labels 45 degrees and align them to the right
# # plt.show()

# ax = within_1s_distribution.set_index('Correct_Classification').plot(
#     kind='bar',
#     stacked=True,
#     figsize=(8, 6),
#     color=['#1f77b4', '#ff7f0e']
# )

# plt.title('Within 1s vs. Outside 1s by Classification Accuracy')
# plt.xlabel('Clip Classified Correctly?')
# plt.ylabel('Percentage')

# # Place legend outside (to the right)
# ax.legend(
#     title='Within 1s of Model-Salient Segment',
#     labels=['Outside 1s', 'Within 1s'],
#     bbox_to_anchor=(1.05, 1),
#     loc='upper left',
#     borderaxespad=0
# )

# plt.xticks(ticks=[0, 1], labels=['Incorrect', 'Correct'], rotation=45, ha='right')
# plt.tight_layout()  # Ensures everything fits nicely
# plt.show()


# # ----------------- Percentage of Participants within 1s by Confidence level -----------------
# # Compute the percentage of participants within 1s per confidence level
# confidence_summary = data.groupby('Confidence_Level')['Within_1s_Margin'].mean().reset_index()
# confidence_summary['Percentage_within_1s'] = confidence_summary['Within_1s_Margin'] * 100

# # Plot the results using seaborn
# plt.figure(figsize=(8, 6))
# sns.barplot(x='Confidence_Level', y='Percentage_within_1s', data=confidence_summary, palette='coolwarm')
# plt.title('Percentage of Participants within 1s by Confidence Level')
# plt.xlabel('Confidence Level')
# plt.ylabel('Percentage Within 1s')
# plt.ylim(0, 100)
# plt.show()

# # ------------------------------------------------------------------------------------------------------
# # Suppose you already have 'within_1s_distribution' something like:
# #   Correct_Classification | Within_1s | Outside_1s
# # 0    False (Incorrect)   |   30.0    |   70.0
# # 1    True  (Correct)     |   60.0    |   40.0

# # 1) Set 'Correct_Classification' as index:
# df_table = within_1s_distribution.set_index('Correct_Classification')[['Within_1s', 'Outside_1s']]

# # 2) Rename the index from True/False to "Correct"/"Incorrect"
# df_table.index = df_table.index.map({False: 'Incorrect', True: 'Correct'})

# # 3) Rename the columns if you like
# df_table.columns = ['Within 1s', 'Outside 1s']

# # 4) Print in a simple text format with float_format for 1 decimal place
# print("=== Within 1s vs. Outside 1s by Classification Accuracy ===")
# print(df_table.to_string(float_format="%.1f"))

