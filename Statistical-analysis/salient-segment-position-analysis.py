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


#------------------ Compute correct classification rate per salient segment position------------------------------
classification_accuracy = data.groupby('Salient_Position_Type')['Correct_Classification'].mean().reset_index()

# Rename column for clarity
classification_accuracy.rename(columns={'Correct_Classification': 'Classification_Accuracy'}, inplace=True)

# Plot bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=classification_accuracy, x='Salient_Position_Type', y='Classification_Accuracy', palette='coolwarm')
plt.title('Correct Classification Rate by Salient Segment Position')
plt.xlabel('Salient Segment Position')
plt.ylabel('Percentage of Participants Classifying Clip Correctly')
plt.ylim(0, 1)  # Normalize y-axis (0 = 0%, 1 = 100%)
plt.xticks(rotation=45)
plt.show()

# ----------------- Miss Rate by Salient Segment Position -----------------
# Group data by Salient Segment Position Type and calculate miss rate
miss_rate_by_position = data.groupby('Salient_Position_Type')['Within_1s_Margin'].apply(lambda x: (x == False).mean()).reset_index()

# Rename column for clarity
miss_rate_by_position.rename(columns={'Within_1s_Margin': 'Miss_Rate'}, inplace=True)

# Plot bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=miss_rate_by_position, x='Salient_Position_Type', y='Miss_Rate', palette='coolwarm')
plt.title('Miss Rate by Salient Segment Position')
plt.xlabel('Salient Segment Position')
plt.ylabel('Percentage of Participants Missing Salient Segment')
plt.ylim(0, 1)  # Normalize y-axis (0 = no misses, 1 = all participants missed)
plt.xticks(rotation=45)
plt.show()