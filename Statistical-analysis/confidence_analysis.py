import pandas as pd
import scipy.stats as stats

# Suppose your DataFrame is 'data' with columns:
#   "Confidence" (numeric: 1=Very Unlikely, 2=Somewhat Unlikely, 3=Somewhat Likely, 4=Very Likely)
#   "Correct_Classification" (boolean: True/False)
# Load data from excel sheet table
data = pd.read_excel('experiment_results.xlsx', sheet_name='table')
confidence_mapping = {'Very Unlikely': 1, 'Somewhat Unlikely': 2, 'Somewhat Likely': 3, 'Very Likely': 4}
data['Confidence'] = data['Confidence_Level'].map(confidence_mapping)

data['Correct_Classification'] = (
    ((data['Clip_Type'].isin(['TP', 'FN'])) & (data['Confidence_Level'].isin(['Somewhat Likely', 'Very Likely']))) |
    ((data['Clip_Type'].isin(['TN', 'FP'])) & (data['Confidence_Level'].isin(['Somewhat Unlikely', 'Very Unlikely'])))
)

# Separate confidence scores by correctness
correct_conf = data.loc[data['Correct_Classification'], 'Confidence']
incorrect_conf = data.loc[~data['Correct_Classification'], 'Confidence']


# Example: Let's do a t-test
t_stat, p_val = stats.ttest_ind(correct_conf, incorrect_conf, equal_var=False)
print("T-test for Confidence by Correctness:")
print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.3f}")
print(f"  mean(correct) = {correct_conf.mean():.2f}, mean(incorrect) = {incorrect_conf.mean():.2f}")

# If data is not normal or you prefer a non-parametric test:
u_stat, p_val_mw = stats.mannwhitneyu(correct_conf, incorrect_conf, alternative='two-sided')
print("\nMann-Whitney U test for Confidence by Correctness:")
print(f"  U-statistic = {u_stat:.3f}, p-value = {p_val_mw:.3f}")
print(f"  median(correct) = {correct_conf.median():.2f}, median(incorrect) = {incorrect_conf.median():.2f}")
