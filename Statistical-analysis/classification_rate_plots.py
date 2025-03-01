import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pingouin as pg
import scipy.stats as st
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve, roc_auc_score

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

data['Correct_Classification'] = (
    ((data['Clip_Type'].isin(['TP', 'FN'])) & (data['Confidence_Level'].isin(['Somewhat Likely', 'Very Likely']))) |
    ((data['Clip_Type'].isin(['TN', 'FP'])) & (data['Confidence_Level'].isin(['Somewhat Unlikely', 'Very Unlikely'])))
)


# MAYBE CONFIDENCE PER CLASSIFICATION TYPE LIKE IF THE PERSON DID TP, TN ,FP ,FN
confidence_map = {
    1: 'Very Unlikely',
    2: 'Somewhat Unlikely',
    3: 'Somewhat Likely',
    4: 'Very Likely'
}


plt.figure(figsize=(10, 5))
sns.boxplot(
    x='Correct_Classification', 
    y='Confidence', 
    data=data, 
    width=0.3, 
    palette='muted'
)
plt.title('Confidence for Correct vs. Incorrect Classifications')
plt.xlabel('Correct Classification?')
plt.ylabel('Confidence Score')
plt.yticks([1, 2, 3, 4], ['Very Unlikely','Somewhat Unlikely','Somewhat Likely','Very Likely'])
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------------------------
# # If Clip_Type in ['TP','FN'], treat it as depressed (1)
# # If Clip_Type in ['FP','TN'], treat it as not depressed (0)
# data['IsDepressed'] = data['Clip_Type'].isin(['TP','FN']).astype(int)

# # y_true = ground truth (0 or 1 for depressed/not depressed)
# y_true = data['IsDepressed'].values  

# # y_score = your numeric confidence
# y_score = data['Confidence'].values

# # Compute ROC curve
# fpr, tpr, thresholds = roc_curve(y_true, y_score)

# # Compute AUC (Area Under the Curve)
# roc_auc = roc_auc_score(y_true, y_score)

# # Plot
# plt.figure(figsize=(6, 6))
# plt.plot(fpr, tpr, color='darkorange',
#          label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # diagonal line
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend(loc="lower right")
# plt.show()


# precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
# pr_auc = auc(recall, precision)

# plt.figure(figsize=(6, 6))
# plt.plot(recall, precision, color='blue',
#          label='Precision-Recall curve (area = %0.2f)' % pr_auc)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc="best")
# plt.show()

# ------------------------------------------------------------------------------------------------------
# Count how many True vs. False
# Data for the pie
num_total = len(data)
num_correct = data['Correct_Classification'].sum()

labels = ['Correct', 'Incorrect']
sizes = [num_correct, num_total - num_correct]

fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
ax.axis('equal')

# Draw a white circle at the center
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

plt.title("Proportion of Correct vs. Incorrect Classifications")
plt.show()
