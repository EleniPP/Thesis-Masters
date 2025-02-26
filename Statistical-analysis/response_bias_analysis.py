import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pingouin as pg
import scipy.stats as stats
import numpy as np

folder = os.path.join(os.path.expanduser("~"), "Downloads")

# Load data from excel sheet table
data = pd.read_excel('experiment_results.xlsx', sheet_name='table')
confidence_mapping = {'Very Unlikely': 1, 'Somewhat Unlikely': 2, 'Somewhat Likely': 3, 'Very Likely': 4}
data['Confidence'] = data['Confidence_Level'].map(confidence_mapping)

plt.figure(figsize=(8, 5))
sns.countplot(x='Selected_Time_Interval', data=data, palette='coolwarm')
plt.title('Distribution of Selected Time Intervals')
plt.xlabel('Time Interval')
plt.ylabel('Count of Selections')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Count the frequency of each interval
observed = data['Selected_Time_Interval'].value_counts().sort_index()
print("Observed counts:\n", observed)

# Expected count: if responses were uniformly distributed,
# each interval would be selected equally often.
n_intervals = len(observed)
expected_count = observed.sum() / n_intervals
expected = [expected_count] * n_intervals

# Perform the Chi-square goodness-of-fit test
chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
print("Chi-square statistic: {:.2f}, p-value: {:.4f}".format(chi2, p))

# 'Participant_ID', 'Clip_ID', and 'Selected_Time_Interval'
# If Selected_Time_Interval is categorical (e.g., "0-1 sec"), consider encoding it as an ordinal variable
# For example, create a new column that maps each interval to a numeric code:
interval_order = {"0-1 sec": 1, "1-2 sec": 2, "2-3 sec": 3, "3-4 sec": 4, "4-5 sec": 5,
                  "5-6 sec": 6, "6-7 sec": 7, "7-8.5 sec": 8}
data['Interval_Code'] = data['Selected_Time_Interval'].map(interval_order)

# Now run repeated measures ANOVA where each participant is measured across different Clip_IDs
rm_results = pg.rm_anova(dv='Interval_Code', within='Clip_ID', subject='Participant_ID', data=data, detailed=True)
print(rm_results)
