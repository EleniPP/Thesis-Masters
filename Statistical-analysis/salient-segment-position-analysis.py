import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pingouin as pg
import scipy.stats as stats
import numpy as np

folder = os.path.join(os.path.expanduser("~"), "Downloads")

# Load data from excel sheet table
data = pd.read_excel('real_experiment_results.xlsx', sheet_name='table')

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

# Apply the classification function to label each row
data['Classification_Type'] = data.apply(get_classification_type, axis=1)
#------------------ Compute correct classification rate per salient segment position------------------------------
# classification_accuracy = data.groupby('Salient_Position_Type')['Correct_Classification'].mean().reset_index()

# # Rename column for clarity
# classification_accuracy.rename(columns={'Correct_Classification': 'Classification_Accuracy'}, inplace=True)

# # Plot bar chart
# plt.figure(figsize=(8, 5))
# ax = sns.barplot(data=classification_accuracy, x='Salient_Position_Type', y='Classification_Accuracy', palette='coolwarm')
# plt.title('Correct Classification Rate by Salient Segment Position')
# plt.xlabel('Salient Segment Position')
# plt.ylabel('Percentage of Participants Classifying Clip Correctly')
# plt.ylim(0, 1)  # Normalize y-axis (0 = 0%, 1 = 100%)
# plt.xticks(rotation=45)
# # Make the bars thinner
# for patch in ax.patches:
#     current_width = patch.get_width()        # get current width
#     new_width = current_width * 0.5            # reduce width by 50%
#     diff = current_width - new_width           # calculate difference
#     patch.set_width(new_width)                 # set new width
#     patch.set_x(patch.get_x() + diff / 2)        # recenter the patch
# plt.tight_layout()
# plt.show()





# ----------------- True Positive Rate by Salient Segment Position -----------------
# Group data by Salient Segment Position Type and compute the proportion of rows where Classification_Type is 'TP'
# Filter the data for rows where Clip_Type is 'TP'
tp_data = data[data['Clip_Type'] == 'TP']

classification_accuracy = tp_data.groupby('Salient_Position_Type').apply(lambda group: (group['Classification_Type'] == 'TP').mean()).reset_index(name='Classification_Accuracy')

# Plot bar chart
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=classification_accuracy, x='Salient_Position_Type', y='Classification_Accuracy', palette='coolwarm')
plt.title('True Positive Rate by Salient Segment Position')
plt.xlabel('Salient Segment Position')
plt.ylabel('Proportion of Clips Classified as TP')
plt.ylim(0, 1)  # Normalize y-axis (0 = 0%, 1 = 100%)
plt.xticks(rotation=45)

# Make the bars thinner
for patch in ax.patches:
    current_width = patch.get_width()
    new_width = current_width * 0.5
    diff = current_width - new_width
    patch.set_width(new_width)
    patch.set_x(patch.get_x() + diff / 2)

plt.tight_layout()
plt.show()









# # ----------------- Miss Rate by Salient Segment Position -----------------
# # Group data by Salient Segment Position Type and calculate miss rate
# miss_rate_by_position = data.groupby('Salient_Position_Type')['Within_1s_Margin'].apply(lambda x: (x == False).mean()).reset_index()

# # Rename column for clarity
# miss_rate_by_position.rename(columns={'Within_1s_Margin': 'Miss_Rate'}, inplace=True)

# # Plot bar chart
# plt.figure(figsize=(8, 5))
# ax = sns.barplot(data=miss_rate_by_position, x='Salient_Position_Type', y='Miss_Rate', palette='coolwarm')
# plt.title('Miss Rate by Salient Segment Position')
# plt.xlabel('Salient Segment Position')
# plt.ylabel('Percentage of Participants Missing Salient Segment')
# plt.ylim(0, 1)  # Normalize y-axis (0 = no misses, 1 = all participants missed)
# plt.xticks(rotation=45)
# for patch in ax.patches:
#     current_width = patch.get_width()        # get current width
#     new_width = current_width * 0.5            # reduce width by 50%
#     diff = current_width - new_width           # calculate difference
#     patch.set_width(new_width)                 # set new width
#     patch.set_x(patch.get_x() + diff / 2) 
# plt.tight_layout() 
# plt.show()