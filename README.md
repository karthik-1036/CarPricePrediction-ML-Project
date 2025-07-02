
# 🚗 Used Car Price Prediction (India-Specific Dataset)

A machine learning project that predicts the resale price of used cars in India based on various specifications like make, model, fuel type, kilometers driven, engine capacity, and more. Built to simulate real-world deployment with a clean architecture, model pipeline, and dynamic Streamlit UI.

---

## 🔍 Problem Statement

Car resale prices vary due to numerous factors such as brand, model, fuel type, mileage, transmission, and more. Accurately predicting the price of a second-hand car can benefit:
- Buyers looking for fair deals 💰
- Dealers setting price benchmarks 📊
- Platforms offering instant car valuations ⚙️

---

## 🎯 Objectives

- Build a regression model to predict resale price of used cars
- Handle categorical, numerical, and textual features efficiently
- Create an interactive frontend using Streamlit
- Prepare for real-world ML deployment (MLOps-ready)

---

## 🧠 Features Used

- Make & Model  
- Year of Manufacture  
- Fuel Type & Transmission  
- Owner Type & Seller Type  
- Max BHP & Torque (with RPMs)  
- Engine Capacity, Dimensions  
- Color, Location, Seating Capacity  
- Kilometers Driven

---

## 📦 Project Structure

```
CarPricePrediction-ML-Project/
│
├── cleaned/                # Cleaned & imputed dataset
├── src/
│   ├── model.py            # Model training & evaluation
│   └── preprocessing.py    # Data cleaning & feature extraction
│
├── app/
│   ├── ui.py               # Streamlit app code
│   └── input_schema.yaml   # Defines app inputs dynamically
│
├── saved_models/           # Trained .pkl model
├── notebooks/              # EDA and exploration notebooks
├── requirements.txt        # Dependencies
└── README.md               # You're here!
```

---

## 🔧 Tech Stack

- **Language**: Python 3  
- **ML Model**: XGBoost Regressor  
- **Preprocessing**: OneHotEncoder, ColumnTransformer  
- **Frontend**: Streamlit  
- **Visualization**: Matplotlib, Seaborn  
- **Model Saving**: joblib  
- **Environment**: Conda

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/karthik-1036/CarPricePrediction-ML-Project.git
cd car-price-predictor

# 2. Create and activate virtual env (recommended)
conda create -n car_price_env python=3.10 -y
conda activate car_price_env
pip install -r requirements.txt

# 3. Train model (optional if model already saved)
python src/model.py

# 4. Launch Streamlit app
streamlit run app/ui.py
```

---

## 📊 Model Performance

- **R² Score (Test)**: `0.87`
- **RMSE**: `₹6.2 Lakhs`
- **MAE**: `₹2.4 Lakhs`
- **Cross-Validation R² Mean**: `0.80+`

---

## 🧪 Future Work

- SHAP-based model explainability
- MLflow experiment tracking
- FastAPI backend for deployment
- Docker + CI/CD integration
- Live data scraping from car platforms (OLX, CarWale)
- Multi-model ensembling for price bands

---

## 🤝 Contributing

Contributions, ideas, or feature requests are welcome. Fork and submit a PR or open an issue.

---

## 📬 Contact

**Author**: Karthik P  
**LinkedIn**: https://www.linkedin.com/in/karthik-prasad-ai/
**Email**: karthikprasad2206@gmail.com
