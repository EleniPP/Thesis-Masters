import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pingouin as pg
import scipy.stats as st
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix

def within_margin(sel_start, sel_end, model_start, model_end, margin=1.0):
    start_close = abs(sel_start - model_start) <= margin
    end_close = abs(sel_end - model_end) <= margin
    return start_close and end_close

folder = os.path.join(os.path.expanduser("~"), "Downloads")

# Load data from excel sheet table
data = pd.read_excel('real_experiment_results.xlsx', sheet_name='table')

data['True_Label'] = data['Clip_Type'].isin(['TP','FN']).astype(int)
data['Human_Prediction'] = (data['Confidence_Level'] > 5).astype(int)

# 3a. Using pandas.crosstab for a quick table:
confusion_table = pd.crosstab(
    data['True_Label'], 
    data['Human_Prediction'], 
    rownames=['Actual'], 
    colnames=['Predicted']
)
print(confusion_table)

# 3b. Or using sklearn's confusion_matrix:
cm = confusion_matrix(data['True_Label'], data['Human_Prediction'])
print(cm)

data['Correct_Classification'] = (
    ((data['Clip_Type'].isin(['TP', 'FN'])) & (data['Confidence_Level'] > 5)) |
    ((data['Clip_Type'].isin(['TN', 'FP'])) & (data['Confidence_Level'] <= 5))
)

# Create a new column with more descriptive labels
data['Classification_Label'] = data['Correct_Classification'].map({True: 'Correct', False: 'Incorrect'})

plt.figure(figsize=(10, 5))
sns.boxplot(
    x='Classification_Label', 
    y='Confidence_Level', 
    data=data, 
    width=0.3, 
    palette='muted'
)
plt.title('Confidence for Correct vs. Incorrect Classifications')
plt.xlabel('Correct Classification?')
plt.ylabel('Confidence Score')
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

# ------------------------------------Frequency of each confidence level-------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you already have your data loaded and processed:
data['Classification_Label'] = data['Correct_Classification'].map({True: 'Correct', False: 'Incorrect'})

# Create a count plot with confidence levels on the x-axis and different bars for Correct vs. Incorrect
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Confidence_Level', hue='Classification_Label', palette='muted')
plt.title('Frequency of Confidence Levels by Classification Outcome')
plt.xlabel('Confidence Level')
plt.ylabel('Number of Participants')
plt.legend(title='Classification', loc='upper right')
plt.tight_layout()
plt.show()
