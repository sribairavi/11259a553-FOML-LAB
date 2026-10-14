import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Load dataset
data = pd.read_csv("heart.csv")

# Separate features and target
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# Convert categorical data to numeric
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Create model and handle missing values
model = make_pipeline(
    SimpleImputer(strategy="median"),
    GradientBoostingClassifier(random_state=1)
)

# Train model
model.fit(X_train, y_train)

# Check accuracy
print("Accuracy:", round(model.score(X_test, y_test), 3))

# Example prediction using the same feature columns
new_data = X_test.iloc[[0]]

prediction = model.predict(new_data)

print("Heart disease? (1=yes, 0=no):", prediction[0])