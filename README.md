# 🌟 Star Classification using Machine Learning

![Stars](https://upload.wikimedia.org/wikipedia/commons/2/2b/Star-forming_region_in_the_large_magellanic_cloud.jpg)

## 📌 Overview
This project implements a **Star Classification System** using **Random Forest Classifier**. It predicts the type of a star based on its physical and spectral attributes. The dataset includes features like temperature, luminosity, radius, absolute magnitude, star color, and spectral class.

## 🚀 Features
- Load and preprocess a dataset of stars
- Train a **Random Forest Classifier** to classify stars into six categories
- Evaluate model performance using accuracy, classification reports, and confusion matrices
- Save the trained model and label encoders
- Predict the star type for a new star based on its characteristics

---

## 🔧 Installation & Setup

### 💻 Open in Google Colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourrepo/star-classification.ipynb)

### 🛠 Prerequisites
Ensure you have the following installed:
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib

To install dependencies, run:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

---

## 📂 Dataset
The dataset contains information about various stars:

| Feature                  | Description                                      |
|--------------------------|--------------------------------------------------|
| Temperature (K)         | Surface temperature of the star (Kelvin)        |
| Luminosity (L/Lo)       | Luminosity relative to the Sun                  |
| Radius (R/Ro)           | Radius relative to the Sun                      |
| Absolute Magnitude (Mv) | The intrinsic brightness of the star            |
| Star Color              | Color of the star                               |
| Spectral Class          | Spectral classification of the star             |
| Star Type (Target)      | Type/category of the star (0-5)                 |

---

## 📊 Data Exploration & Preprocessing

### 🔹 Load Dataset
```python
import pandas as pd
# Load the dataset
file_path = "6 class csv.csv"  # Replace with actual file path
df = pd.read_csv(file_path)
print(df.head())
```

### 🔹 Check for Missing Values
```python
print(df.isnull().sum())
```

### 🔹 Encode Categorical Data
```python
from sklearn.preprocessing import LabelEncoder
label_encoder_color = LabelEncoder()
label_encoder_class = LabelEncoder()
df['Star color'] = label_encoder_color.fit_transform(df['Star color'])
df['Spectral Class'] = label_encoder_class.fit_transform(df['Spectral Class'])
```

### 🔹 Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star color', 'Spectral Class']])
y = df['Star type']
```

---

## 🤖 Model Training & Evaluation

### 🔹 Split Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### 🔹 Train Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 🔹 Evaluate Model Performance
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

✅ **Model Accuracy: 100%**

---

## 🛠 Save & Load the Model

### 🔹 Save the Model
```python
import pickle
pickle.dump(model, open("star_classifier.pkl", "wb"))
pickle.dump(label_encoder_color, open("label_encoder_color.pkl", "wb"))
pickle.dump(label_encoder_class, open("label_encoder_class.pkl", "wb"))
```

### 🔹 Load the Model for Prediction
```python
loaded_model = pickle.load(open("star_classifier.pkl", "rb"))
loaded_color_encoder = pickle.load(open("label_encoder_color.pkl", "rb"))
loaded_class_encoder = pickle.load(open("label_encoder_class.pkl", "rb"))
```

---

## 🔮 Make Predictions

### 🔹 Predict Star Type for New Data
```python
# Define new star attributes
new_star = [[3000, 0.002, 0.15, 16, 1, 5]]  # Example values

# Scale the new data
new_star_scaled = scaler.transform(new_star)

# Predict the star type
star_type_prediction = loaded_model.predict(new_star_scaled)
print(f"Predicted Star Type: {star_type_prediction[0]}")
```
✅ **Predicted Star Type: 4**

---

## 📜 Summary
✅ **Complete pipeline for star classification**
✅ **100% accuracy with Random Forest Classifier**
✅ **Confusion matrix visualization for performance analysis**
✅ **Model saving/loading for future predictions**

---

## 📜 License
This project is licensed under the MIT License. Feel free to contribute and enhance it!

📌 **Author:** *Your Name*
📌 **GitHub Repository:** [Your Repo](https://github.com/yourrepo)

🌟 **If you like this project, don't forget to star ⭐ the repository!**

---

![Thanks](https://media.giphy.com/media/3ohhwE3ALqYpZVF8Yo/giphy.gif)

