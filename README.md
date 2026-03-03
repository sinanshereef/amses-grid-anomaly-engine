# ⚡ AMSES — Smart Energy Intelligence Platform  

AI-Powered Meter Surveillance & Energy Anomaly Detection System  
Built using **Random Forest Classifier (92% Accuracy)** with advanced feature engineering and interactive analytics.

---

## 🚀 Overview

**AMSES (Anomaly and Meter Surveillance in Electricity Systems)** is an intelligent energy monitoring platform that detects abnormal electricity consumption patterns in smart meters.

The system analyzes:

- Property & meter information  
- Environmental conditions  
- Energy consumption patterns  
- Electrical health parameters  

It automatically engineers backend features and predicts anomaly types with confidence scoring and visual explanations.

---

## 🎯 Key Features

✔ AI-based anomaly detection (Random Forest)  
✔ 5-class anomaly classification  
✔ Automatic backend feature engineering  
✔ Interactive analytics dashboard  
✔ Confidence gauge visualization  
✔ Probability distribution chart  
✔ Energy distribution donut chart  
✔ Radar feature analysis  
✔ Risk scoring system  
✔ Recommended action engine  

---

## 🧠 Anomaly Classes

| Code | Anomaly Type |
|------|-------------|
| 0 | ✅ Normal – No Anomaly |
| 1 | ⚠️ Meter Bypass / Tampering |
| 2 | 🔶 Unusual Consumption Pattern |
| 3 | 🌩️ Grid Outage Impact |
| 4 | 🔴 Overload / High Usage |

---

## 🏗️ System Architecture

```
User Input → Feature Engineering → Scaling → Random Forest Model
            → Probability Output → Visualization Layer → Action Engine
```

---

## 📊 Engineered Features (Auto-Computed)

The backend automatically derives intelligent features such as:

- Energy Deviation Ratio  
- Energy per Occupant  
- Peak-to-Total Ratio  
- Load Utilization %  
- Energy per Sq.ft  
- Season Encoding  
- Month & Day Encoding  
- Composite Anomaly Risk Score  

These features significantly improve model accuracy and anomaly detection reliability.

---

## 🖥️ Streamlit UI Preview

```markdown
<img width="1919" height="980" alt="image" src="https://github.com/user-attachments/assets/ca22cace-51e4-4122-b2c0-c29ecae30733" />

<img width="1919" height="980" alt="image" src="https://github.com/user-attachments/assets/32c3291b-467f-4e5d-a46b-8d3acde5572a" />

<img width="1919" height="983" alt="image" src="https://github.com/user-attachments/assets/a5ffe5a6-f52b-4d3c-ad76-0f5cf793fb9d" />

<img width="1919" height="973" alt="image" src="https://github.com/user-attachments/assets/f76e8cb0-b79d-461a-9d55-f4d04437969c" />
```

## ⚙️ Installation

```bash
git clone https://github.com/your-username/amses-smart-energy-intelligence.git
cd amses-smart-energy-intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit numpy pandas scikit-learn joblib plotly
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Make sure these files are present in the same directory:

```
anomaly_model.pkl
scaler.pkl
app.py
```

---

## 📈 Model Information

- Algorithm: Random Forest Classifier  
- Accuracy: 92%  
- Multi-Class Classification  
- Feature Scaling: StandardScaler  
- Trained on Kerala Smart Meter dataset  

---

## 📊 Visual Analytics Included

### 1️⃣ Confidence Gauge
Displays model prediction confidence level.

### 2️⃣ Probability Bar Chart
Shows probability distribution across all 5 anomaly classes.

### 3️⃣ Energy Distribution Donut
Peak vs Off-Peak vs Remaining usage.

### 4️⃣ Radar Feature Analysis
Compares engineered features against baseline.

### 5️⃣ Key Indicator Cards
Highlights:

- Energy deviation %
- Load utilization %
- Power factor
- Risk score

---

## 🔧 Recommended Action Engine

Based on detected anomaly, the system automatically suggests:

- Field inspection  
- Energy audit  
- Outage compensation review  
- Overload safety check  
- Or no action required  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Scikit-Learn  
- NumPy  
- Pandas  
- Plotly  

---

## 📁 Project Structure

```
├── app.py
├── anomaly_model.pkl
├── scaler.pkl
├── electricity.ipynb
├── requirements.txt
└── assets/
    └── streamlit_ui.png
```

---

## 💡 Why This Project Is Powerful

This is not just a prediction model.

It is:

- A full AI application  
- With real-time analytics  
- Risk interpretation  
- Visual explanation  
- Action recommendation system  

This demonstrates:

- ML model deployment  
- Feature engineering expertise  
- Production-ready UI development  
- Business-oriented AI thinking  

---

## 📌 Future Improvements

- Real-time IoT meter integration  
- API deployment (FastAPI)  
- Cloud hosting (AWS / Azure)  
- Role-based dashboard  
- Automated anomaly reporting system  

---

## 👨‍💻 Author

**Sinan Shereef**  
AI/ML Engineer | Data Scientist  

---

## ⭐ If You Like This Project

Give it a star ⭐ on GitHub and connect with me for collaboration!
