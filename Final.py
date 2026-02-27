# Step 2: Data Exploration 
# 1. Load the dataset

# Needed libraries
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

df = pd.read_csv("team13_realestate.csv")
print(df.head()) # Ensure it works

# 2. Explore data structure and types
# Size + Column Names
print("\nDataset Shape:", df.shape)
print("Column Names:", df.columns)

# Column types and non-null counts
print("\nDataset Info:")
df.info()

# See the number of non-missing vaules, the mean, standard deviation, smallest value, first quartile, median, third quartile, and largest values
print("\nDescribe:")
print(df.describe())

# See how many missing values in each column
print("\nMissing Values:")
print(df.isna().sum())

# 3. Identify data quality issues
# Everything was already standardized, there are some null values but only of a small percentage. To handle missing values in square foot, we imputed median square feet and because price is a key variable, we removed rows that are missing that value. There aren't any invalid values
# 4. Document findings

# Step 3: Data Cleaning  (make sure you record and present your assumptions and decisions)
# 1. Handle missing values
# 2. Fix data type issues
# 3. Address inconsistencies
# 4. Validate cleaned data

# Create a copy to preserve original dataset
df_clean = df.copy()

# Calculate median for square_feet
sqft_median = df_clean['square_feet'].median()

# Fill missing square_feet values with median
df_clean['square_feet'] = df_clean['square_feet'].fillna(sqft_median)

# Drop rows where price is missing (price is essential)
df_clean = df_clean.dropna(subset=['price'])

print("\nAfter Handling Missing Values:")
df_clean.info()

# Make sure there are no impossible values, confirm there aren't any missing values, and that the shape is correct

# Impossible values
print("\nChecking for Impossible Values:")

print("Price <= 0:", (df_clean['price'] <= 0).sum())
print("Square Feet <= 0:", (df_clean['square_feet'] <= 0).sum())
print("Negative Age:", (df_clean['age_years'] < 0).sum())
print("Days on Market <= 0:", (df_clean['days_on_market'] <= 0).sum())

# Remove the impossible values if they exist
df_clean = df_clean[df_clean['price'] > 0]
df_clean = df_clean[df_clean['square_feet'] > 0]
df_clean = df_clean[df_clean['age_years'] >= 0]
df_clean = df_clean[df_clean['days_on_market'] > 0]

print("\nDataset Shape After Removing Impossible Values:", df_clean.shape)

# Detect Outliers
Q1 = df_clean['price'].quantile(0.25)
Q3 = df_clean['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[
    (df_clean['price'] < lower_bound) |
    (df_clean['price'] > upper_bound)
]

print("\nNumber of Price Outliers Detected:", len(outliers))

# We're detecting outliers but not removing them since they apply in real-life real estate

# Confirming it works
print("\nFinal Cleaned Dataset Info:")
df_clean.info()

print("\nFinal Cleaned Dataset Shape:", df_clean.shape)

print("\nFinal Summary Statistics:")
print(df_clean.describe())

# Step 4: Analysis 
# 1. Answer each of the 6 questions assigned to your team.
# 2. Perform statistical analysis
# 3. Create calculated metrics
# 4. Document insights

# Step 5: Visualization
# 1. Create 4 meaningful visualizations
# 2. Add proper labels and titles
# 3. Save plots
# 4. Interpret results

# %% [markdown]
# ## Questions:

# %% [markdown]
# ### Question 1
# outlines price per square foot by location
# %%
print(df_clean.groupby('location').apply(lambda x:(x['price'].div(x['square_feet'])).mean()))

# %% [markdown]
# ### Question 2
# Correlation effect of age on price is a very weak negative correlation meaning price goes down as age increases since the result is negative and between 0 and -0.199

# %%
age_price_corr = df_clean['age_years'].corr(df_clean['price'])
print(age_price_corr)

plt.scatter(df_clean['age_years'],df_clean['price'], alpha=0.6,s=5)
plt.xlabel("Age (Years)")
plt.ylabel("Price")
plt.title("Plotting Price to age ")
plt.grid(True, linestyle='--',alpha=0.4)
plt.tight_layout()
plt.savefig("age_vs_price.png")
plt.show()

# %% [markdown]
# ### Question 3
# This box plot shows and compares the days on market by property type and shows that houses stay on the market for a comparably long time compared to other property types

# %%
df_clean.boxplot(column='days_on_market',by='property_type')
plt.xlabel("Propety Type")
plt.ylabel("Days on Market")
plt.title("Days on Market by Location")
plt.suptitle("")
plt.tight_layout()
plt.savefig("Days_by_Location.png")
plt.show()

# %% [markdown]
# ### Question 4

# 1. Bedroom vs Price
# There doesn't seem to be a strong correlation between bedrooms and price. On the scatterplot created, listing prices for properties don't increase or decrease with the number of bedrooms the houses have. For a 1 bedroom house, listing prices could range from $50k to $1M, and for a 3 bedroom house, it could be from $50k to 900k, and the listing prices for a 2 bedroom house are higher than for a 5 bedroom house.

bedroom_price_corr = df_clean['bedrooms'].corr(df_clean['price'])
print(bedroom_price_corr)

# Create scatterplot, fix the size for h and w
plt.figure(figsize=(8,5)) 
plt.scatter(df_clean['bedrooms'], df_clean['price'], alpha=0.6)

# Create labels
plt.xlabel("Number of Bedrooms")
plt.ylabel("Listing Price ($)")
plt.title("Relationship Between Bedrooms and Listing Price")

# Grid lines
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# Save the plot and display the figure
plt.savefig("bedrooms_vs_price.png")
plt.show()

# %% [markdown]
# ### Question 5
# Prices are a lot higher and there are far more outliers for properties with garages than there are for those without them. The median price for properties with garages is lower than for those without garages, and first quartiles are the exact same. It seems there is some correlation between them--houses with garages are generally preferred over those without garages, but price and garage availability aren't that closely tied

# 2. Garage vs Price
# Create boxplot of certain size
plt.figure(figsize=(8, 5))
df_clean.boxplot(column='price', by='garage')

# Label
plt.xlabel("Garage (False = No, True = Yes)")
plt.ylabel("Listing Price ($)")
plt.title("Price Distribution Based on Garage Availability")

# Clean the title since 'by' skews it a little
plt.suptitle("")
plt.tight_layout()

# Save + Display
plt.savefig("garage_vs_price.png")
plt.show()

garage_price_corr = df_clean['garage'].corr(df_clean['price'])
print(garage_price_corr)

# %% [markdown]
# ### Question 6
# How many properties were sold before 30 days (%)
percentage_sold_30 = (
    df_clean.groupby('location')
    .apply(lambda x: ((x['sold']) & (x['days_on_market'] <= 30)).mean() * 100)
)

print(percentage_sold_30)

# Approximately 16–18% of properties across all locations sell within 30 days. Urban areas have the slightly highest percentage at 17.98%, followed by suburban areas at 17.75%, while rural areas have the lowest at 16.67%. The differences are minimal though, suggesting that location does not strongly impact the likelihood of a property selling within 30 days in this dataset.
