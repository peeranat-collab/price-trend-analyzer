import streamlit as st
import pandas as pd
from app import load_data   # ใช้ฟังก์ชันเดิม

# -------------------------
# Helper
# -------------------------
def get_price(df, material, year, month):
    row = df[
        (df["วัสดุ"] == material) &
        (df["ปี"] == year) &
        (df["เดือน"] == month)
    ]
    if len(row) == 0:
        return None
    return row["ราคา/หน่วย"].mean()

# =========================
# Page
# =========================
st.title("📊 วิเคราะห์ต้นทุน (YoY Impact Analysis)")

df = load_data()

if len(df) == 0:
    st.warning("ยังไม่มีข้อมูลราคาในระบบ")
    st.stop()

# ---- เลือกช่วง ----
st.subheader("1️⃣ เลือกช่วงเวลา")

col1, col2 = st.columns(2)
with col1:
    sel_month = st.selectbox("เลือกเดือน", range(1,13))
with col2:
    sel_year = st.selectbox(
        "เลือกปี (ปีนี้)",
        sorted(df["ปี"].unique(), reverse=True)
    )

base_year = sel_year - 1
st.caption(f"เปรียบเทียบ {sel_month}/{sel_year} กับ {sel_month}/{base_year}")

# ---- สัดส่วน ----
st.subheader("2️⃣ โครงสร้างต้นทุน (%)")

materials = [
    "น้ำมันดีเซล",
    "อะลูมิเนียม",
    "ผ้าฝ้าย (Cotton)",
    "เม็ดพลาสติก PET",
    "ค่าแรง"
]

weights = {}
cols = st.columns(len(materials))
for i, m in enumerate(materials):
    with cols[i]:
        weights[m] = st.number_input(m, 0.0, 100.0, 0.0)

total_weight = sum(weights.values())
if total_weight == 0:
    st.warning("กรุณาใส่สัดส่วน")
    st.stop()

# ---- คำนวณ ----
rows = []
for m in materials:
    p_now = get_price(df, m, sel_year, sel_month)
    p_prev = get_price(df, m, base_year, sel_month)

    if p_now is None or p_prev is None:
        continue

    yoy = (p_now - p_prev) / p_prev * 100
    impact = yoy * (weights[m] / total_weight)

    rows.append({
        "วัสดุ": m,
        "YoY %": round(yoy,2),
        "Impact (%)": round(impact,2)
    })

result_df = pd.DataFrame(rows)
st.dataframe(result_df, use_container_width=True)

total_impact = result_df["Impact (%)"].sum()
st.markdown("---")

if total_impact > 0:
    st.error(f"🔺 ต้นทุนรวมเพิ่ม ~{total_impact:.2f}%")
else:
    st.success(f"🔻 ต้นทุนรวมลด ~{total_impact:.2f}%")
