import streamlit as st
import pickle
import numpy as np
import pandas as pd

# โหลดโมเดลและ scaler
with open("customer_churn_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# ตั้งค่าหน้าเว็บ
st.title("Customer Churn Prediction")
st.write("ป้อนข้อมูลของลูกค้าเพื่อดูว่ามีโอกาสเลิกใช้บริการหรือไม่")

# สร้างอินพุตสำหรับผู้ใช้
complain = st.selectbox("ลูกค้าเคยร้องเรียนหรือไม่?", [0, 1])
age = st.number_input("อายุของลูกค้า", min_value=18, max_value=100, value=30)
is_active = st.selectbox("ลูกค้าเป็น Active Member หรือไม่?", [0, 1])
num_products = st.slider("จำนวนผลิตภัณฑ์ที่ใช้ (เช่น บัญชีเงินฝาก, บัตรเครดิต, สินเชื่อ ฯลฯ)", 1, 4, 1)
geography = st.selectbox("ประเทศของลูกค้า", ["France", "Germany", "Spain"])
balance = st.number_input("ยอดเงินในบัญชี", min_value=0.0, max_value=500000.0, value=50000.0)

# แปลงค่าหมวดหมู่เป็นตัวเลข
geo_map = {"France": 0, "Germany": 1, "Spain": 2}
geography = geo_map[geography]

# สร้าง DataFrame สำหรับโมเดล
input_df = pd.DataFrame([[complain, age, is_active, num_products, geography, balance]], 
                        columns=["Complain", "Age", "IsActiveMember", "NumOfProducts", "Geography", "Balance"])

# ปรับขนาดข้อมูลด้วย StandardScaler
input_data = scaler.transform(input_df)

# ทำการทำนาย
if st.button("🔮 ทำนายผลลัพธ์"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1][0]
    
    if prediction[0] == 1:
        st.error(f"🚨 ลูกค้ารายนี้มีแนวโน้มที่จะเลิกใช้บริการ ({probability:.2%} ความน่าจะเป็น)")
    else:
        st.success(f"✅ ลูกค้ารายนี้มีแนวโน้มที่จะอยู่ต่อ ({(1 - probability):.2%} ความน่าจะเป็น)")

# แสดงข้อความแนะนำ
st.markdown("**เคล็ดลับ:** ปรับข้อมูลแต่ละตัวเพื่อดูว่าปัจจัยไหนส่งผลต่อการเลิกใช้บริการมากที่สุด!")