from sklearn.datasets import load_iris                  # the flower dataset
from sklearn.model_selection import train_test_split    # tool that splits data
from sklearn.neighbors import KNeighborsClassifier      # the KNN model

X, y = load_iris(return_X_y=True)                        # X = measurements, y = species number

X_train, X_test, y_train, y_test = train_test_split(     # split into learn + exam
    X, y, test_size=0.2, random_state=1)                 # 20% is the hidden exam

model = KNeighborsClassifier(n_neighbors=5)              # KNN: ask the 5 nearest flowers
model.fit(X_train, y_train)                              # learn from the training flowers

print("Accuracy:", model.score(X_test, y_test))