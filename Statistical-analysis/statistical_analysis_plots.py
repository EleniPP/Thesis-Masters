import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pingouin as pg
import scipy.stats as st
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


# Melt the DataFrame to gather all influential features into one column
data_filtered = data[data["Within_1s_Margin"] == True]
print(data_filtered)
data_melted = data_filtered.melt(id_vars=['Clip_Type'], 
                         value_vars=['Influential_Features-Eyebrows', 
                                     'Influential_Features-Eyes', 
                                     'Influential_Features-Mouth'],
                         var_name='Feature_Type', 
                         value_name='Influential_Feature')

# Ignore empty or placeholder values ('-' or NaN) but keep the row
data_melted = data_melted[data_melted['Influential_Feature'].notna()]
data_melted = data_melted[data_melted['Influential_Feature'] != '-']

# Rename 'Feature_Type' values for better readability
data_melted['Feature_Type'] = data_melted['Feature_Type'].str.replace('Influential_Features-', '')

# **NEW STEP: Split multi-feature selections into separate rows**
data_melted = data_melted.assign(Influential_Feature=data_melted['Influential_Feature'].str.split('; '))
data_melted = data_melted.explode('Influential_Feature')

# Split into three DataFrames
df_eyebrows = data_melted[data_melted['Feature_Type'] == 'Eyebrows']
df_eyes = data_melted[data_melted['Feature_Type'] == 'Eyes']
df_mouth = data_melted[data_melted['Feature_Type'] == 'Mouth']

# Set up subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# Define colors
colors = {'Eyebrows': 'purple', 'Eyes': 'blue', 'Mouth': 'orange'}

# Plot for Eyebrows
sns.countplot(data=df_eyebrows, x='Clip_Type', hue='Influential_Feature', palette='Purples', ax=axes[0])
axes[0].set_title('Influential Eyebrow Features by Clip Type')
axes[0].set_ylabel('Feature Count')
axes[0].legend(title='Eyebrow Features', bbox_to_anchor=(1, 1))

# Plot for Eyes
sns.countplot(data=df_eyes, x='Clip_Type', hue='Influential_Feature', palette='Blues', ax=axes[1])
axes[1].set_title('Influential Eye Features by Clip Type')
axes[1].set_ylabel('Feature Count')
axes[1].legend(title='Eye Features', bbox_to_anchor=(1, 1))

# Plot for Mouth
sns.countplot(data=df_mouth, x='Clip_Type', hue='Influential_Feature', palette='Oranges', ax=axes[2])
axes[2].set_title('Influential Mouth Features by Clip Type')
axes[2].set_xlabel('Clip Classification')
axes[2].set_ylabel('Feature Count')
axes[2].legend(title='Mouth Features', bbox_to_anchor=(1, 1))

# Show the plots
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

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

# ----------------- 6. Misclassification & Confidence Analysis -----------------
# TO INVESTIGATE
# MAYBE CONFIDENCE PER CLASSIFICATION TYPE LIKE IF THE PERSON DID TP, TN ,FP ,FN
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



# -----------------------Lets see if I can have slider so Ill have points --------------------
# # Example: selected time intervals (in seconds)
# selected_times = np.array([1.2, 1.5, 1.7, 2.0, 2.1, 2.3, 2.4, 2.3, 1.9, 2.0, 1.8, 2.2])

# # Compute the KDE of the selected time intervals
# kde = st.gaussian_kde(selected_times)

# # Create an array of time values over the range of your data
# x_vals = np.linspace(selected_times.min(), selected_times.max(), 200)
# density = kde(x_vals)

# # Find the mode: the x value corresponding to the maximum density
# mode_index = np.argmax(density)
# mode_value = x_vals[mode_index]

# # Alternatively, you can compute the weighted mean (expected value)
# weighted_mean = np.sum(x_vals * density) / np.sum(density)

# # Plot the density and the representative point(s)
# plt.figure(figsize=(10,5))
# sns.kdeplot(selected_times, shade=True, label="Density")
# plt.plot(mode_value, density[mode_index], 'ro', label=f'Mode: {mode_value:.2f}s')
# plt.plot(weighted_mean, kde(weighted_mean), 'go', label=f'Weighted Mean: {weighted_mean:.2f}s')
# plt.xlabel("Time Interval (s)")
# plt.ylabel("Density")
# plt.legend()
# plt.title("Density of Selected Time Intervals with Representative Points")
# plt.show()

# ----------------- Random but its the training loss in the final run -----------------
# losses = [
#     0.2926458000741994,
#     0.2743494115595969,
#     0.2639775844904349,
#     0.2563856609774633,
#     0.2502011940214906,
#     0.2451521453287426,
#     0.2407373493898019,
#     0.2366191045227293,
#     0.23362763501845404,
#     0.2304643115299483,
#     0.22800213583280365,
#     0.22565840763573905,
#     0.22376067434880953,
#     0.22202685685767734,
#     0.2201857603143075,
#     0.2186396135849986,
#     0.21762811369650792,
#     0.21617758288177874,
#     0.21534885124756514,
#     0.21441180242271352,
#     0.2134273900884846,
#     0.21265532253609698,
#     0.21162983254128623,
#     0.21134096703368652,
#     0.21046173437990734,
#     0.20986428085673958,
#     0.20928652222934066,
#     0.20893149894857832,
#     0.20825088368435857,
#     0.2079418288328803
# ]

# # Create the plot
# plt.figure(figsize=(8, 5))
# plt.plot(losses, marker='o', linestyle='-', color='olivedrab')
# plt.title("Training Losses per Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.tight_layout()
# plt.show()









# # 0.2926458000741994
# # 0.2743494115595969
# # 0.2639775844904349
# # 0.2563856609774633
# # 0.2502011940214906
# # 0.2451521453287426
# # 0.2407373493898019
# # 0.2366191045227293
# # 0.23362763501845404
# # 0.2304643115299483
# # 0.22800213583280365
# # 0.22565840763573905
# # 0.22376067434880953
# # 0.22202685685767734
# # 0.2201857603143075
# # 0.2186396135849986
# # 0.21762811369650792
# # 0.21617758288177874
# # 0.21534885124756514
# # 0.21441180242271352
# # 0.2134273900884846
# # 0.21265532253609698
# # 0.21162983254128623
# # 0.21134096703368652
# # 0.21046173437990734
# # 0.20986428085673958
# # 0.20928652222934066
# # 0.20893149894857832
# # 0.20825088368435857
# # 0.2079418288328803