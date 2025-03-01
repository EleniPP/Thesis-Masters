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
# Define correct classification based on salient segment type and participant response
data['Correct_Classification'] = (
    ((data['Clip_Type'].isin(['TP', 'FN'])) & (data['Confidence_Level'].isin(['Somewhat Likely', 'Very Likely']))) |
    ((data['Clip_Type'].isin(['TN', 'FP'])) & (data['Confidence_Level'].isin(['Somewhat Unlikely', 'Very Unlikely'])))
)


# ----------------- Percentage of Participants Within 1s by Classification Accuracy -----------------
# Compute proportions of participants within and outside 1s for correct and incorrect classifications
within_1s_distribution = data.groupby('Correct_Classification')['Within_1s_Margin'].value_counts(normalize=True).unstack().reset_index()

# Rename columns
within_1s_distribution.rename(columns={True: 'Within_1s', False: 'Outside_1s'}, inplace=True)

# Convert to percentage
within_1s_distribution[['Within_1s', 'Outside_1s']] *= 100

# Plot stacked bar chart
within_1s_distribution.set_index('Correct_Classification').plot(kind='bar', stacked=True, figsize=(8, 6), color=['#ce434a', '#48a389'])
# plt.title('Within 1s vs. Outside 1s by Classification Accuracy')
# plt.xlabel('Clip Classified Correctly?')
# plt.ylabel('Percentage')
# plt.legend(
#     title='Within 1s of Model-Salient Segment',
#     labels=['Outside 1s', 'Within 1s'],
#     bbox_to_anchor=(1.05, 1),  # 5% to the right of the axes, aligned at top
#     loc='upper left',
#     borderaxespad=0.
# )
# plt.xticks(ticks=[0, 1], labels=['Incorrect', 'Correct'])
# plt.xticks(rotation=45, ha='right')  # Rotate labels 45 degrees and align them to the right
# plt.show()

ax = within_1s_distribution.set_index('Correct_Classification').plot(
    kind='bar',
    stacked=True,
    figsize=(8, 6),
    color=['#1f77b4', '#ff7f0e']
)

plt.title('Within 1s vs. Outside 1s by Classification Accuracy')
plt.xlabel('Clip Classified Correctly?')
plt.ylabel('Percentage')

# Place legend outside (to the right)
ax.legend(
    title='Within 1s of Model-Salient Segment',
    labels=['Outside 1s', 'Within 1s'],
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0
)

plt.xticks(ticks=[0, 1], labels=['Incorrect', 'Correct'], rotation=45, ha='right')
plt.tight_layout()  # Ensures everything fits nicely
plt.show()


# ----------------- Percentage of Participants within 1s by Confidence level -----------------
# Compute the percentage of participants within 1s per confidence level
confidence_summary = data.groupby('Confidence_Level')['Within_1s_Margin'].mean().reset_index()
confidence_summary['Percentage_within_1s'] = confidence_summary['Within_1s_Margin'] * 100

# Plot the results using seaborn
plt.figure(figsize=(8, 6))
sns.barplot(x='Confidence_Level', y='Percentage_within_1s', data=confidence_summary, palette='coolwarm')
plt.title('Percentage of Participants within 1s by Confidence Level')
plt.xlabel('Confidence Level')
plt.ylabel('Percentage Within 1s')
plt.ylim(0, 100)
plt.show()

# ------------------------------------------------------------------------------------------------------
# Suppose you already have 'within_1s_distribution' something like:
#   Correct_Classification | Within_1s | Outside_1s
# 0    False (Incorrect)   |   30.0    |   70.0
# 1    True  (Correct)     |   60.0    |   40.0

# 1) Set 'Correct_Classification' as index:
df_table = within_1s_distribution.set_index('Correct_Classification')[['Within_1s', 'Outside_1s']]

# 2) Rename the index from True/False to "Correct"/"Incorrect"
df_table.index = df_table.index.map({False: 'Incorrect', True: 'Correct'})

# 3) Rename the columns if you like
df_table.columns = ['Within 1s', 'Outside 1s']

# 4) Print in a simple text format with float_format for 1 decimal place
print("=== Within 1s vs. Outside 1s by Classification Accuracy ===")
print(df_table.to_string(float_format="%.1f"))
