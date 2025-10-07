# Project-004-Titanic-survival-prediction-with-preprocessing-pipeline

A machine learning project to predict passenger survival on the Titanic using Python, pandas, and scikit-learn. This project includes data preprocessing, feature engineering, pipeline creation, model training, evaluation, and saving the trained pipeline for reuse.

---

## Dataset

The dataset used is the [Titanic dataset](https://www.kaggle.com/c/titanic/data) from Kaggle. It contains information about Titanic passengers, including:

- PassengerId
- Survived (target)
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

---

## Project Steps

### 1. Load Libraries
Imported all necessary Python libraries for data handling, preprocessing, modeling, and evaluation, including:

- pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, preprocessing, RandomForest)
- joblib for saving the trained model

### 2. Load Dataset
Uploaded `Titanic-Dataset.csv` to Colab and loaded it into a pandas DataFrame.

### 3. Explore & Understand Data (EDA)
- Checked dataset shape and column data types.
- Inspected missing values.
- Examined basic statistics of numeric features.
- Listed all columns for reference.

### 4. Feature Engineering
Created new features to improve model performance:

- `Title` – extracted from passenger names
- `FamilySize` – SibSp + Parch + 1
- `IsAlone` – 1 if FamilySize = 1, else 0
- `CabinLetter` – first letter of Cabin
- `FarePerPerson` – Fare divided by FamilySize

### 5. Preprocessing + Pipeline
- Numeric features: filled missing values with median, scaled using StandardScaler.
- Categorical features: filled missing values with most frequent value, one-hot encoded.
- Combined preprocessing using `ColumnTransformer`.
- Built a full pipeline including preprocessing and RandomForestClassifier.

### 6. Train-Test Split + Model Training
- Split dataset into 80% training and 20% validation.
- Trained the pipeline on the training set.
- Predicted on validation set.
- Evaluated using accuracy, classification report, and confusion matrix.

### 7. Cross-Validation
Performed 5-fold cross-validation to check model stability and compute average performance.

### 8. Save Model and Metrics
- Saved the trained pipeline using `joblib`.
- Saved validation metrics to a CSV file.

---

## How to Use

1. Clone the repository.
2. Upload `Titanic-Dataset.csv` to your environment (e.g., Colab).
3. Install required packages:

```bash
pip install pandas scikit-learn joblib
````

4. Run the notebook/script to train the model and generate predictions.
5. Load the saved model for future predictions:

```python
import joblib

pipeline = joblib.load('titanic_pipeline_model.joblib')
predictions = pipeline.predict(new_data)
```

---

## Results

* Validation Accuracy: ~ 0.7877
* Cross-Validation Mean Accuracy: ~  0.8058

---

## Libraries Used

* pandas
* numpy
* scikit-learn
* joblib

---

## Author

**Olawale Samuel Olaitan**
Bachelor of Science in Electronics and Computer Engineering
[GitHub Profile](https://github.com/Olanle)

---
