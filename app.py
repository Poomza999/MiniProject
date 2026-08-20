import streamlit as st
import joblib
import numpy as np

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Currency Trend Predictor", page_icon="📈")
st.title("📈 ระบบพยากรณ์แนวโน้มค่าเงินบาท (THB)")

# 2. ฟังก์ชันโหลดโมเดล (ใช้ Cache เพื่อให้เว็บโหลดเร็วขึ้น)
@st.cache_resource
def load_model():
    return joblib.load('currency_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ ไม่พบไฟล์ 'currency_model.pkl' กรุณาตรวจสอบในโฟลเดอร์")
    st.stop()

# 3. สร้าง UI สำหรับรับค่า Input
st.write("กรอกข้อมูลปัจจุบันเพื่อพยากรณ์แนวโน้มอัตราแลกเปลี่ยนในวันพรุ่งนี้")

col1, col2 = st.columns(2)
with col1:
    current_val = st.number_input("อัตราแลกเปลี่ยนวันนี้", min_value=0.0, format="%.4f")
with col2:
    ma_val = st.number_input("ค่าเฉลี่ย 7 วันย้อนหลัง (MA_7)", min_value=0.0, format="%.4f")

# 4. ปุ่มประมวลผลและแสดงผลลัพธ์
if st.button("ประมวลผล (Predict)", type="primary"):
    # แปลงข้อมูลให้อยู่ในรูปแบบ Array 2 มิติ สำหรับส่งให้โมเดล
    features = np.array([[current_val, ma_val]])
    prediction = model.predict(features)[0]
    
    st.markdown("---")
    if prediction == 1:
        st.success("🔼 **แนวโน้มพรุ่งนี้:** ค่าเงินมีโอกาสปรับตัว **สูงขึ้น (Up Trend)**")
    else:
        st.error("🔽 **แนวโน้มพรุ่งนี้:** ค่าเงินมีโอกาสปรับตัว **ลดลง (Down Trend)**")