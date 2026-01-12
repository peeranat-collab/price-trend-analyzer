import streamlit as st
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="วิเคราะห์แนวโน้มราคา", layout="wide")

# =========================
# ตั้งค่า
# =========================
DATA_FILE = "data.csv"

products = [
    "กระเป๋า Delivery ใบเล็ก",
    "กระเป๋า Delivery ใบใหญ่",
    "แจ็คเก็ต Delivery"
]

materials_base = [
    "เม็ดพลาสติก",
    "ผ้าคัทตอน",
    "เหล็ก",
    "ค่าแรง",
    "ค่าขนส่ง"
]

# =========================
# Utility Functions
# =========================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=[
            "สินค้า", "เดือน", "ปี", "วัสดุ",
            "ราคา/หน่วย", "ปริมาณ", "ต้นทุน",
            "overhead_percent", "timestamp"
        ])

def save_data(df):
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")

def yoy_compare(df, selected_month, selected_year):
    current = df[(df["เดือน"] == selected_month) & (df["ปี"] == selected_year)]
    prev = df[(df["เดือน"] == selected_month) & (df["ปี"] == selected_year - 1)]

    cur_sum = current.groupby("วัสดุ")["ต้นทุน"].sum()
    prev_sum = prev.groupby("วัสดุ")["ต้นทุน"].sum()

    result = pd.DataFrame({
        "ปีที่แล้ว": prev_sum,
        "ปีนี้": cur_sum
    }).fillna(0)

    result["% เปลี่ยนแปลง"] = ((result["ปีนี้"] - result["ปีที่แล้ว"]) / result["ปีที่แล้ว"].replace(0, 1)) * 100
    return result.reset_index()

# โหลดข้อมูล
df_data = load_data()

# =========================
# Sidebar
# =========================
st.sidebar.title("📊 เมนู")
menu = st.sidebar.radio("เลือกเมนู", [
    "Dashboard",
    "กรอกข้อมูลต้นทุน",
    "ตารางข้อมูล",
    "วิเคราะห์แนวโน้ม",
    "Export"
])

# =========================
# Dashboard
# =========================
if menu == "Dashboard":
    st.title("📊 Dashboard")

    if len(df_data) == 0:
        st.info("ยังไม่มีข้อมูล กรุณากรอกข้อมูลก่อน")
    else:
        st.subheader("ข้อมูลล่าสุด")
        st.dataframe(df_data.tail(10), use_container_width=True)

        st.subheader("ต้นทุนรวมตามสินค้า")
        summary = df_data.groupby("สินค้า")["ต้นทุน"].sum()
        st.bar_chart(summary)

# =========================
# กรอกข้อมูล
# =========================
elif menu == "กรอกข้อมูลต้นทุน":
    st.title("➕ กรอกข้อมูลต้นทุน (ละเอียดระดับวัสดุ)")

    col1, col2, col3 = st.columns(3)
    with col1:
        product = st.selectbox("เลือกสินค้า", products)
    with col2:
        month = st.selectbox("เดือน", list(range(1, 13)))
    with col3:
        year = st.selectbox("ปี", list(range(2023, 2031)))

    st.subheader("เลือกวัสดุที่ใช้")
    selected_materials = st.multiselect(
        "เลือกวัสดุ",
        materials_base + ["วัสดุอื่นๆ"]
    )

    overhead_percent = st.number_input("Overhead (%)", min_value=0.0, step=1.0)

    material_rows = []

    st.markdown("---")

    for mat in selected_materials:
        st.markdown(f"### {mat}")
        c1, c2 = st.columns(2)
        with c1:
            price = st.number_input(f"ราคา/หน่วย ({mat})", min_value=0.0, step=1.0, key=f"p_{mat}")
        with c2:
            qty = st.number_input(f"ปริมาณที่ใช้ ({mat})", min_value=0.0, step=0.1, key=f"q_{mat}")

        cost = price * qty

        material_rows.append({
            "สินค้า": product,
            "เดือน": month,
            "ปี": year,
            "วัสดุ": mat,
            "ราคา/หน่วย": price,
            "ปริมาณ": qty,
            "ต้นทุน": cost,
            "overhead_percent": overhead_percent,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    if len(material_rows) > 0:
        df_preview = pd.DataFrame(material_rows)
        base_total = df_preview["ต้นทุน"].sum()
        overhead_value = base_total * (overhead_percent / 100)
        final_total = base_total + overhead_value

        st.markdown("---")
        st.subheader("สรุป")
        st.write(f"ต้นทุนรวมวัสดุ: {base_total:.2f} บาท")
        st.write(f"Overhead: {overhead_value:.2f} บาท")
        st.success(f"ต้นทุนรวมต่อสินค้า = {final_total:.2f} บาท")

        if st.button("บันทึกข้อมูล"):
            new_df = pd.DataFrame(material_rows)
            df_all = pd.concat([df_data, new_df], ignore_index=True)
            save_data(df_all)
            st.success("บันทึกข้อมูลเรียบร้อยแล้ว 🎉")
            st.experimental_rerun()

# =========================
# ตารางข้อมูล
# =========================
elif menu == "ตารางข้อมูล":
    st.title("📋 ตารางข้อมูล")

    if len(df_data) == 0:
        st.info("ยังไม่มีข้อมูล")
    else:
        st.dataframe(df_data, use_container_width=True)

        st.subheader("แนวโน้มต้นทุนรวม (ต่อวัสดุ)")
        pivot = df_data.groupby(["ปี", "เดือน", "วัสดุ"])["ต้นทุน"].sum().reset_index()
        pivot["เวลา"] = pivot["ปี"].astype(str) + "-" + pivot["เดือน"].astype(str)

        chart_df = pivot.pivot(index="เวลา", columns="วัสดุ", values="ต้นทุน")
        st.line_chart(chart_df)

# =========================
# วิเคราะห์แนวโน้ม (NEW)
# =========================
elif menu == "วิเคราะ
