import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
imputer = joblib.load(os.path.join(BASE_DIR, "imputer.pkl"))


import streamlit.components.v1 as components
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add sparkling effect
components.html("""
<style>
@keyframes sparkle {
  0% { transform: scale(0); opacity: 1; }
  100% { transform: scale(1); opacity: 0; }
}
.sparkle {
  position: fixed;
  width: 10px;
  height: 10px;
  background-color: gold;
  border-radius: 50%;
  pointer-events: none;
  animation: sparkle 0.8s linear forwards;
  z-index: 9999;
}
</style>
<script>
const script = document.createElement('script');
script.innerHTML = `
document.addEventListener('mousemove', function(e) {
  var sparkle = document.createElement('div');
  sparkle.classList.add('sparkle');
  
  sparkle.style.left = (e.pageX - 5) + 'px';
  sparkle.style.top = (e.pageY - 5) + 'px';
  sparkle.style.backgroundColor = 'hsl(' + Math.random() * 360 + ', 100%, 50%)';
  
  // Create style element if it doesn't exist
  if (!document.getElementById('sparkle-style')) {
      var style = document.createElement('style');
      style.id = 'sparkle-style';
      style.innerHTML = \`
        @keyframes sparkle {
          0% { transform: scale(0); opacity: 1; }
          100% { transform: scale(1); opacity: 0; }
        }
        .sparkle {
          position: fixed;
          width: 10px;
          height: 10px;
          background-color: gold;
          border-radius: 50%;
          pointer-events: none;
          animation: sparkle 0.8s linear forwards;
          z-index: 9999;
        }
      \`;
      document.head.appendChild(style);
  }
  
  document.body.appendChild(sparkle);
  
  setTimeout(function() {
    sparkle.remove();
  }, 800);
});
`;
window.parent.document.body.appendChild(script);
</script>
""", height=0, width=0)

st.markdown("""
    <style>
    .glitter-container {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .glitter-text {
        background: linear-gradient(to right, #660000, #ff0000, #990000, #660000);
        background-size: 200% auto;
        color: #000;
        background-clip: text;
        text-fill-color: transparent;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
    }
    .beating-heart {
        display: inline-block;
        animation: heartbeat 2.5s infinite;
    }
    @keyframes shine {
        to {
            background-position: 200% center;
        }
    }
    @keyframes heartbeat {
        0% { transform: scale(1); }
        15% { transform: scale(1.3); }
        30% { transform: scale(1); }
        45% { transform: scale(1.15); }
        60% { transform: scale(1); }
        100% { transform: scale(1); }
    }
    </style>
    <div class="glitter-container">
        <span class="beating-heart">🫀</span> <span class="glitter-text">Heart Disease Prediction App</span>
    </div>
""", unsafe_allow_html=True)
st.write("Enter patient details below:")
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate (thalach)", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [1, 2, 3])
if st.button("Predict"):
    new_patient = np.array([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]])
    new_patient = imputer.transform(new_patient)
    prediction = model.predict(new_patient)
    probability = model.predict_proba(new_patient)[0][1] * 100
    
    st.subheader("Prediction Results")
    
    if prediction[0] == 1:
        st.error(f"⚠️ Heart Disease Detected (Probability: {probability:.1f}%)")
    else:
        st.success(f"✅ No Heart Disease Detected (Probability: {probability:.1f}%)")
    
    st.write("### Risk Analysis")
    st.progress(int(probability))
    
    risk_level = "Low"
    if probability >= 70:
        risk_level = "High"
    elif probability >= 30:
        risk_level = "Moderate"
        
    risk_data = pd.DataFrame({
        "Metric": ["Heart Disease Probability", "Risk Level"],
        "Value": [f"{probability:.2f}%", risk_level]
    })
    
    st.table(risk_data)
    

