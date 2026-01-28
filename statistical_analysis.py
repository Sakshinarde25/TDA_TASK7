# Statistical Business Analysis
# Beginner-friendly & Error-free

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Load data
try:
    df = pd.read_csv("business_data.csv")
except FileNotFoundError:
    print("❌ business_data.csv not found")
    exit()

# -----------------------------
# 1️⃣ DESCRIPTIVE STATISTICS
# -----------------------------
mean_sales = df["Sales"].mean()
median_sales = df["Sales"].median()
std_sales = df["Sales"].std()

print("DESCRIPTIVE STATISTICS")
print("Mean Sales:", mean_sales)
print("Median Sales:", median_sales)
print("Standard Deviation:", std_sales)

# -----------------------------
# 2️⃣ DISTRIBUTION ANALYSIS
# -----------------------------
plt.hist(df["Sales"], bins=5)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# Normality Test
shapiro_test = stats.shapiro(df["Sales"])
print("\nNormality Test (Shapiro-Wilk):", shapiro_test)

# -----------------------------
# 3️⃣ CORRELATION ANALYSIS
# -----------------------------
correlation = df["Sales"].corr(df["Marketing_Spend"])
print("\nCorrelation (Sales vs Marketing):", correlation)

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 4️⃣ HYPOTHESIS TESTING
# -----------------------------

# Hypothesis 1: One-sample t-test
t_test = stats.ttest_1samp(df["Sales"], 45000)

# Hypothesis 2: Independent t-test
high_marketing = df[df["Marketing_Spend"] > 9000]["Sales"]
low_marketing = df[df["Marketing_Spend"] <= 9000]["Sales"]
ind_t_test = stats.ttest_ind(high_marketing, low_marketing)

# Hypothesis 3: ANOVA
anova_test = stats.f_oneway(high_marketing, low_marketing)

# Save results
with open("hypothesis_tests_results.txt", "w") as f:
    f.write("One Sample T-Test:\n" + str(t_test) + "\n\n")
    f.write("Independent T-Test:\n" + str(ind_t_test) + "\n\n")
    f.write("ANOVA Test:\n" + str(anova_test))

print("\nHypothesis Tests Completed")

# -----------------------------
# 5️⃣ CONFIDENCE INTERVAL
# -----------------------------
confidence_level = 0.95
n = len(df["Sales"])
mean = mean_sales
std = std_sales
z = stats.norm.ppf((1 + confidence_level) / 2)

margin_error = z * (std / np.sqrt(n))
ci_lower = mean - margin_error
ci_upper = mean + margin_error

print("\n95% Confidence Interval:")
print(f"{ci_lower:.2f} to {ci_upper:.2f}")

# -----------------------------
# 6️⃣ REGRESSION ANALYSIS
# -----------------------------
X = df["Marketing_Spend"]
Y = df["Sales"]

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()

print("\nREGRESSION SUMMARY")
print(model.summary())

print("\n✅ Statistical Analysis Completed Successfully")
