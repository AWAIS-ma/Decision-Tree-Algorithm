import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("diabetes.csv")

data.drop_duplicates(inplace=True)      
data.dropna(inplace=True)  

le_gender = LabelEncoder()
le_smoking = LabelEncoder()
data['gender'] = le_gender.fit_transform(data['gender'])
data['smoking_history'] = le_smoking.fit_transform(data['smoking_history'])

X = data.drop("diabetes", axis=1)
y = data["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
labels = ['Non-Diabetic', 'Diabetic']
sns.heatmap(matrix, annot=True, fmt='d', cmap='Reds' , xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()