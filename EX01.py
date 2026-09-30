import pandas as pd
import seaborn as sns


data = sns.load_dataset("titanic")

# Select required columns
data = data[["survived", "pclass", "sex", "age", "fare", "embarked"]]

# Check missing values before cleaning
print("Missing values before cleaning:")
print(data.isnull().sum())

# Fill missing age values with median
data["age"] = data["age"].fillna(data["age"].median())

# Fill missing embarked values with the most frequent value
data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])

# Convert categorical columns into numerical columns
data = pd.get_dummies(
    data,
    columns=["sex", "embarked"],
    dtype=int
)

# Display first 5 rows
print("\nData after preprocessing:")
print(data.head())

# Check for remaining missing values
print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nAny missing values left?", data.isnull().values.any())

new = pd.DataFrame([{"pclass":2,"sex":"female","age":27,"fare":30,"embarked":"C"}])  # one new passenger
new = pd.get_dummies(new, columns=["sex","embarked"], dtype=int)     # clean it the same way
new = new.reindex(columns=data.drop("survived", axis=1).columns, fill_value=0)  # match the columns
print(new)
