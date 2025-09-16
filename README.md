

# 🏡 California House Price Prediction

This project predicts **median house prices in California districts** using the **California Housing dataset**.
It applies data preprocessing with Scikit-learn pipelines, custom feature engineering, and a **Random Forest Regression** model.

---

## 📂 Project Structure

```
california-house-price-prediction/
│
├── notebooks/
│   └── California_Housing_Project.ipynb   # Jupyter notebook with EDA & model training
│
├── src/
│   ├── custom_transformer.py              # Custom feature engineering transformer
│   ├── house_price_predictor.py           # Script for loading model & making predictions
│   └── __init__.py                        # (optional, package initializer)
│
├── models/
│   ├── final_random_forest_model.pkl      # Trained Random Forest model
│   └── full_pipeline.pkl                  # Preprocessing pipeline
│
├── requirements.txt                       # Python dependencies
│
├── README.md                              # Project documentation
│
└── .gitignore                             # Ignore cache & unnecessary files
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/UvaishLunsheth/california-house-price-prediction.git
cd california-house-price-prediction
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the notebook

Open the Jupyter notebook for **EDA, training, and evaluation**:

```
notebooks/California_Housing_Project.ipynb
```

### Run prediction script

You can load the model and make predictions:

```bash
python src/house_price_predictor.py
```

Or directly in Python:

```python
import joblib

# Load model & preprocessing pipeline
model = joblib.load("models/final_random_forest_model.pkl")
pipeline = joblib.load("models/full_pipeline.pkl")

# Example input (features after preprocessing)
# Replace with real values
sample_input = [[8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]]
prediction = model.predict(sample_input)

print("Predicted Median House Price:", prediction)
```

---

## 📊 Dataset

This project uses the **California Housing dataset** available in `scikit-learn`:

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
```

---

## 📈 Results

* **Model:** Random Forest Regressor (Tuned with RandomizedSearchCV)
* **Evaluation Metrics:**

  * R² Score: *\~49237.21*
  * RMSE: *\~0.814*



---

## 🔮 Future Work

* Experiment with **Gradient Boosting, XGBoost, CatBoost**
* Perform **Hyperparameter tuning**
* Deploy using **Flask / FastAPI / Streamlit**
* Build a **web dashboard** for predictions

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork this repo and submit a pull request.

---

⚠️ Note: The trained Random Forest model (.pkl) is too large to upload to GitHub.  
If you want to use it, please run the notebook to retrain the model, or replace it with your own trained model.


## 📜 License

This project is for **educational purposes** and follows the MIT License.

---

