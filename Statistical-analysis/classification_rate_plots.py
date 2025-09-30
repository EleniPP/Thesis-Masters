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
import krippendorff

def within_margin(sel_start, sel_end, model_start, model_end, margin=1.0):
    start_close = abs(sel_start - model_start) <= margin
    end_close = abs(sel_end - model_end) <= margin
    return start_close and end_close

def gwet_ac1(ratings_df: pd.DataFrame) -> float:
    """
    Compute Gwet’s AC1 for nominal ratings (0/1, etc.), allowing NaNs.
    Each row = one item (clip), each column = one rater’s label.
    """
    data = ratings_df.values
    n_items, m = data.shape

    # Observed agreement Ao
    agree_sum = 0
    total_pairs = 0
    for row in data:
        # ignore NaNs when counting
        vals, counts = np.unique(row[~np.isnan(row)], return_counts=True)
        agree_sum   += sum(c * (c - 1) for c in counts)
        valid_m     = len(row[~np.isnan(row)])
        total_pairs += valid_m * (valid_m - 1)

    Ao = agree_sum / total_pairs

    # Expected agreement Ae
    flat          = data[~np.isnan(data)].flatten()
    vals, counts  = np.unique(flat, return_counts=True)
    p             = counts / flat.size
    Ae            = sum(p_j * (1 - p_j) for p_j in p)

    return (Ao - Ae) / (1 - Ae)

folder = os.path.join(os.path.expanduser("~"), "Downloads")

# Load data from excel sheet table
data = pd.read_excel('real_experiment_results.xlsx', sheet_name='table')

data['Human_Prediction'] = (data['Confidence_Level'] > 5).astype(int)
data['True_Label'] = data['Clip_Type'].isin(['TP','FN']).astype(int)
grouped = data.groupby('Clip_ID')['Human_Prediction'].apply(list).reset_index(name='Ratings')



# Find the maximum number of raters per clip
max_raters = grouped['Ratings'].apply(len).max()

# Build a rectangular array with np.nan for missing
matrix = np.full((len(grouped), max_raters), np.nan, dtype=float)

for i, ratings in enumerate(grouped['Ratings']):
    matrix[i, :len(ratings)] = ratings

# Convert to DataFrame (one column per “rater slot”)
col_names    = [f"rater_{j+1}" for j in range(max_raters)]
clip_ratings = pd.DataFrame(matrix, columns=col_names)

# Compute and print Gwet’s AC1
ac1_value = gwet_ac1(clip_ratings)
print(f"Gwet’s AC1 = {ac1_value:.3f}")

