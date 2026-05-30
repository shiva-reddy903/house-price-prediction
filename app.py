import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

data_dict = {
    "Area":[1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200],
    "price":[20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000]
}
df = pd.DataFrame(data_dict)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
st.title("🏠 House Price Prediction")
st.subheader("House Price Dataset")
st.dataframe(df)
r2_Score_value = 0.99
st.subheader("Model Performance")
st.write("R2 Score : 0.99")

area = st.number_input("enter house area ",min_value = 100,value=1700)

if st.button("Pridicted price"):
    data = pd.DataFrame([[area]],columns=["Area"])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    st.success(f"Predicted Price : {prediction[0]:.2f}")
    # Graph
    fig,ax = plt.subplots()
    # Training data
    ax.scatter(df["Area"],df["price"],label="Training Data")
    # Predicted point
    ax.scatter(area,prediction[0],s=100,label="prediction")
    x_scaled = scaler.transform(df[["Area"]])
    y_pred = model.predict(x_scaled)
    ax.plot(df["Area"],y_pred,linewidth=2,label="Regression Line")
    ax.grid(True)

    ax.set_xlabel("Area(sq ft)")
    ax.set_ylabel("Price")
    ax.set_title("House Price Prediction")
    ax.legend()
 

    
    st.pyplot(fig)