import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pingouin as pg
import scipy.stats as stats
import numpy as np
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
import patsy
import statsmodels.formula.api as smf
import statsmodels.api as sm

def parse_interval(interval_str):
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

# Compute the absolute error in seconds
data["Error"] = (data["Selected_Timestamp"] - data["Model_Mid"]).abs()

data = data[(data["Clip_Type"] == "TP") & (data["Classification_Type"] == "TP")]
# 1. Parse the 'Model_Salient_Interval' into numeric start/end and compute midpoint
intervals = data['Model_Salient_Interval'].str.extract(
    r'(?P<start>\d+\.\d+)-(?P<end>\d+\.\d+)'
).astype(float)
data['salient_time'] = intervals.mean(axis=1)

# Compute 'time_before' = how many seconds before the actual salient moment
data['time_before'] = data['salient_time'] - data['Selected_Timestamp']

# Assign into 1-second bins: 0–1s, 1–2s, …, 7–8s before
bin_edges  = np.arange(0, 9, 1)
bin_labels = [f"{i}-{i+1}s" for i in range(8)]
data['bin'] = pd.cut(
    data['time_before'],
    bins=bin_edges,
    labels=bin_labels,
    right=False
)

# Observed counts per bin: C_b
C = data['bin'].value_counts().sort_index().reindex(bin_labels, fill_value=0)

# Compute exposure A_b: total available “seconds × participants” per bin
# Build a per-clip summary: number of participants & salient_time
clips = (
    data
    .groupby('Clip_ID')
    .agg(
        num_participants=('Participant_ID', 'nunique'),
        salient_time=('salient_time', 'first')
    )
    .reset_index()
)

def available_duration(ts, b_start, b_end, clip_len=8.5):
    win_start, win_end = ts - b_end, ts - b_start
    inter_start = max(0, win_start)
    inter_end   = min(clip_len, win_end)
    return max(0, inter_end - inter_start)

rows = []
for _, clip in clips.iterrows():
    for i, label in enumerate(bin_labels):
        dur = available_duration(clip['salient_time'], i, i+1)
        rows.append({
            'bin': label,
            'available_time': dur * clip['num_participants']
        })
A_df = pd.DataFrame(rows)
A = A_df.groupby('bin')['available_time'].sum().reindex(bin_labels, fill_value=0)

rates = C / A

# Fit a Poisson GLM with offset = log(A_b)
glm_data = pd.DataFrame({'count': C, 'A': A}).reset_index().rename(columns={'index': 'bin'})
# --- build the glm_data DataFrame as before ---
glm_data = pd.DataFrame({
    'bin': C.index.tolist(),
    'count': C.values,
    'A':     A.values
})

# Drop any bins that had zero exposure
glm_data = glm_data[glm_data['A'] > 0].copy()

# Cast dtypes explicitly
glm_data['count'] = glm_data['count'].astype(int)
glm_data['A']     = glm_data['A'].astype(float)

# Create design matrix
dummies = pd.get_dummies(glm_data['bin'], drop_first=True)
X = sm.add_constant(dummies).astype(float)         # ensure float
y = glm_data['count'].values                       # numpy array of ints

# Fit the Poisson GLM with the log-A offset
glm = sm.GLM(
    y,
    X,
    offset=np.log(glm_data['A'].values),
    family=sm.families.Poisson()
)
result = glm.fit()
print(result.summary())


# Observed counts per bin: C_b
C = data['bin'].value_counts().sort_index().reindex(bin_labels, fill_value=0)

# vailable‐time per bin: A_b
#    (built from your clips summary)
A = A_df.groupby('bin')['available_time'].sum().reindex(bin_labels, fill_value=0)

summary = pd.DataFrame({
    'count (C_b)':   C.astype(int),
    'exposure (A_b)': A.astype(float),
    'rate (C_b/A_b)': (C/A).round(4)
})

print(summary)






# ----------------------------------



# # 3) Group by Clip_ID (or however you identify each clip)
# error_stats = data.groupby("Clip_ID").agg(
#     mean_error = ("Error", "mean"),
#     std_error = ("Error", "std"),
#     within_1s_pct = ("Within_1s_Margin", lambda x: 100 * x.mean())
# ).reset_index()

# # # 4) Print or display the table
# # print("=== Error Analysis by Clip ===")
# # print(error_stats.to_string(index=False, float_format="%.2f"))

# # Optional: rename columns for clarity or format the output
# # e.g. rename "within_1s_pct" to "% Within 1s"
# error_stats.rename(columns={"within_1s_pct": "%_Within_1s"}, inplace=True)

# print("\n=== Error Analysis by Clip (Final) ===")
# print(error_stats.to_string(index=False, float_format="%.2f"))

# # ----------------- Above table but not per clip. Global analysis -----------------
# # Compute overall summary statistics:
# global_mean_error = data["Error"].mean()
# global_std_error = data["Error"].std()
# global_median_error = data["Error"].median()
# global_pct_within1 = 100 * data["Within_1s_Margin"].mean()

# print("Global Average Error (s):", global_mean_error)
# print("Global Error STD (s):", global_std_error)
# print("Global Median Error (s):", global_median_error)
# print("Global % Within 1s:", global_pct_within1)

# # Create a summary table (by clip or overall)
# global_summary = pd.DataFrame({
#     "Metric": ["Mean Error (s)", "Std Error (s)", "Median Error (s)", "% Within 1s"],
#     "Global Value": [global_mean_error, global_std_error, global_median_error, global_pct_within1]
# })
# print(global_summary)





# # ----------------- Violin plot for Raw error instead of the Box plot for MAE -----------------
# data["RawError"] = data["Selected_Timestamp"] - data["Model_Mid"]

# tp_tp_data = data[(data["Clip_Type"] == "TP") & (data["Classification_Type"] == "TP")]
# tp_fn_data = data[(data["Clip_Type"] == "FN") & (data["Classification_Type"] == "TP")]
# tn_tn_data = data[(data["Clip_Type"] == "TN") & (data["Classification_Type"] == "TN")]
# tn_fp_data = data[(data["Clip_Type"] == "FP") & (data["Classification_Type"] == "TN")]

# # tp_data = data[(data["Clip_Type"] == "TN") & (data["Classification_Type"] == "TN")]

# # tp_data["RawError"] = tp_data["Selected_Timestamp"] - tp_data["Model_Mid"]

# # Determine the global minimum and maximum of RawError across all datasets
# global_min = min(df["RawError"].min() for df in [tp_tp_data, tp_fn_data, tn_tn_data, tn_fp_data])
# global_max = max(df["RawError"].max() for df in [tp_tp_data, tp_fn_data, tn_tn_data, tn_fp_data])

# print(global_min, global_max)

# plt.figure(figsize=(8, 6))
# # Plot a vertical violin plot: the y-axis is RawError.
# sns.violinplot(y="RawError", data=tn_fp_data, color="skyblue")
# plt.ylim(global_min-2, global_max+4)
# # Add a horizontal line at y=0 (the model’s midpoint)
# plt.axhline(y=0, color="red", linestyle="--", label="Model Midpoint")

# plt.ylabel("Raw Error (seconds)")
# plt.title("Vertical Violin Plot of Raw Error")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # ----------------------------------------------------------------------------------------------------
# # ----------------------------------------------------------------------------------------------------
# tn_tn_data = data[
#     (data["Clip_Type"] == "TN") & 
#     (data["Classification_Type"] == "TN")
# ]
# tp_tp_data = data[
#     (data["Clip_Type"] == "TP") & 
#     (data["Classification_Type"] == "TP")
# ]
# fn_tp_data = data[
#     (data["Clip_Type"] == "FN") & 
#     (data["Classification_Type"] == "TP")
# ]
# fp_tn_data = data[
#     (data["Clip_Type"] == "FP") & 
#     (data["Classification_Type"] == "TN")
# ]
# # Filter data only to segments classified as TP by the model
# tp_segments = data[data["Clip_Type"] == "TP"]

# # Create the contingency table using Classification_Label
# tp_table = pd.crosstab(tp_segments["Classification_Label"], tp_segments["Within_1s_Margin"], normalize='index')

# # Rename columns clearly
# tp_table.columns = ['Outside 1s', 'Within 1s']

# # Convert to percentages and reorder
# tp_table = tp_table[['Within 1s', 'Outside 1s']] * 100

# print(tp_table.round(1))



# # -------------------------------------------------------
# # 1) Compute “seconds before” the model midpoint
# data = tp_tp_data.copy()

# data["TimeBefore"] = data["Model_Mid"] - data["Selected_Timestamp"]

# # 2) Keep only those clicks that happened before the midpoint
# before = data[data["TimeBefore"] > 0].copy()

# # 5) Create the binary flag for “clicked 1–3 s before”
# data["clicked_1_3s"] = (
#     data["TimeBefore"].between(1.0, 3.0, inclusive="both")
# ).astype(int)

# # 6) Subset/rename into df_long
# df_long = data.rename(columns={
#     "Participant_ID": "participant_id",
#     "Clip_ID":         "clip_id",
#     "Clip_Type":       "clip_type",
#     "Confidence_Level":"confidence_level",
#     "Salient_Position_Type": "salient_pos_type"
# })[[
#     "participant_id",
#     "clip_id",
#     "clip_type",
#     "confidence_level",
#     "TimeBefore",
#     "clicked_1_3s",
#     "salient_pos_type"
# ]]

# # 7) Inspect
# print(df_long.head())

# from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

# # y: clicked_1_3s, X: intercept column, Z: grouping design
# endog = df_long["clicked_1_3s"]
# exog = sm.add_constant(pd.DataFrame({"intercept": 1}, index=df_long.index))
# # random intercept for participant and clip
# vc = {
#     "participant": "0 + C(participant_id)",
#     "clip":        "0 + C(clip_id)"
# }

# model = BinomialBayesMixedGLM(endog, exog, vc_formula=vc, exog_vc=None)
# res = model.fit_vb()  # variational Bayes for speed
# print(res.summary())





# # 3) Define your bins: 0–1s, 1–2s, …, 7–8s, 8–8.5s
# bin_edges  = list(np.arange(0, 8.5, 1.0)) + [8.5]
# bin_labels = [f"{int(bin_edges[i])}–{bin_edges[i+1]:.1f}s" 
#               for i in range(len(bin_edges)-1)]

# # 4) Bin the TimeBefore values
# before["BeforeBin"] = pd.cut(
#     before["TimeBefore"],
#     bins=bin_edges,
#     labels=bin_labels,
#     include_lowest=True
# )

# # 5) Count, per clip, how many selections fell into each “before” bin
# per_clip_before = (
#     before
#     .groupby(["Clip_ID", "BeforeBin"])
#     .size()
#     .unstack(fill_value=0)
# )

# print("\n=== Counts of timestamps BEFORE the midpoint, per clip ===")
# print(per_clip_before)

# # 6) If you also want the grand total across all clips:
# grand_totals_before = per_clip_before.sum(axis=0)
# print("\n=== Grand total counts (all clips) for each BEFORE-bin ===")
# print(grand_totals_before)


# # Compute percentages:
# percentages = grand_totals_before / grand_totals_before.sum() * 100

# # Round nicely and display:
# print(percentages.round(1))


# # ---------------------------------------------------------
# # Statistical analysis


# # define your bin edges / labels again
# bin_edges  = list(np.arange(0, 8.5, 1.0)) + [8.5]
# bin_labels = [f"{int(bin_edges[i])}–{bin_edges[i+1]:.1f}s"
#               for i in range(len(bin_edges)-1)]

# # 1) count clicks per bin (as before)
# click_counts = (
#     before
#     .groupby("BeforeBin")
#     .size()
#     .reindex(bin_labels, fill_value=0)
# )

# # 2) compute exposures per bin
# # for each bin, count how many rows in `data` *could* have clicked there:
# exposures = {}
# for b, (start, end) in zip(bin_labels, zip(bin_edges, bin_edges[1:])):
#     # a rating "could" fall in bin b if Model_Mid >= end
#     # (and of course the participant rated that clip)
#     exposures[b] = data.loc[data["Model_Mid"] >= end].shape[0]

# exposures = pd.Series(exposures)

# # 3) compute rates
# rates = (click_counts / exposures) * 100   # % of possible clicks that fell in that bin

# # 4) show them side by side
# result = pd.DataFrame({
#     "#Clicks":   click_counts,
#     "#Exposures": exposures,
#     "Rate (%)":  rates.round(1)
# })
# print(result)

# -------------------------------------------------
# # 1) Aggregate your clicks into counts per (participant, clip, bin)
# agg = (
#     before
#     .groupby(["Participant_ID","Clip_ID","BeforeBin"])
#     .size()
#     .reset_index(name="Count")
# )

# # 2) Build your fixed‐effect design (intercept + BeforeBin dummies)
# exog = patsy.dmatrix("1 + C(BeforeBin)", agg, return_type="dataframe")

# # 3) Build your random‐effect designs
# #    one column per participant, one per clip (all 0/1 dummies)
# re_part  = patsy.dmatrix("0 + C(Participant_ID)", agg, return_type="dataframe")
# re_clip  = patsy.dmatrix("0 + C(Clip_ID)",       agg, return_type="dataframe")

# exog_vc = pd.concat([re_part, re_clip], axis=1)

# # 2) Build the `ident` array
# n_part = re_part.shape[1]    # number of participant columns
# n_clip = re_clip.shape[1]    # number of clip columns
# ident  = np.array([0]*n_part + [1]*n_clip)

# # 3) Fit the model, passing ident
# model = PoissonBayesMixedGLM(
#     endog    = agg["Count"].values,
#     exog     = exog.values,
#     exog_vc  = exog_vc.values,
#     ident    = ident
# )
# fit = model.fit_vb()  # or .fit_map()
# print(fit.summary())


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

