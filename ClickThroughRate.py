from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 0
data = pd.read_csv('advertising.csv')
print(data.head())

# Step 1
print(data.isnull().sum())
print(data.columns)

# Step 2
x = data.iloc[:, 0:7]
x = x.drop(['Ad Topic Line', 'City'], axis=1)
y = data.iloc[:, 9]

# Step 3
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=4)

# Step 4
Lr = LogisticRegression(C=0.01, random_state=0)
Lr.fit(x_train, y_train)

# Step 5
y_pred = Lr.predict(x_test)
print(y_pred)

# Step 6
y_pred_proba = Lr.predict_proba(x_test)
print(y_pred_proba)

print(accuracy_score(y_test, y_pred))

print(f1_score(y_test, y_pred))
