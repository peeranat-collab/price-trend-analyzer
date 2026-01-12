import streamlit as st
import pandas as pd

st.set_page_config(page_title="วิเคราะห์แนวโน้มราคา", layout="wide")

# -----------------------
# ค่าเริ่มต้น
# -----------------------
products = [
    "กระเป๋า Delivery ใบเล็ก",
    "กระเป๋า Delivery ใบใหญ่",
    "แจ็คเก็ต Delivery"
]

materials = [
    "เม็ดพลาสติก",
    "ผ้าคัทตอน",
    "เหล็ก",
    "ค่าแรง",
    "ค่าขนส่ง"
]

if "data" not in st.session_state:
    st.session_state.data = []

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("📊 เมนู")

menu = st.sidebar.radio(
    "เลือกเมนู",
    [
        "Dashboard",
        "กรอกข้อมูลต้นทุน",
        "ตารางข้อมูล",
        "Export"
    ]
)

# -----------------------
# Dashboard
# -----------------------
if menu == "Dashboard":
    st.title("📊 Dashboard")

    if len(st.session_state.data) == 0:
        st.info("ยังไม่มีข้อมูล กรุณากรอกข้อมูลก่อน")
    else:
        df = pd.DataFrame(st.session_state.data)

        st.subheader("ข้อมูลล่าสุด")
        st.dataframe(df.tail(5), use_container_width=True)

        st.subheader("ต้นทุนรวมตามสินค้า")
        summary = df.groupby("สินค้า")["ต้นทุนรวม"].sum()
        st.bar_chart(summary)

# -----------------------
# กรอกข้อมูล
# -----------------------
elif menu == "กรอกข้อมูลต้นทุน":
    st.title("➕ กรอกข้อมูลต้นทุน")

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
        materials + ["วัสดุอื่นๆ"]
    )

    total_cost = 0
    detail = []

    for mat in selected_materials:
        st.markdown(f"### {mat}")
        c1, c2 = st.columns(2)
        with c1:
            price = st.number_input(f"ราคา {mat}", min_value=0.0, step=1.0, key=f"p_{mat}")
        with c2:
            qty = st.number_input(f"ปริมาณที่ใช้ {mat}", min_value=0.0, step=0.1, key=f"q_{mat}")

        cost = price * qty
        total_cost += cost
        detail.append(f"{mat}: {price} x {qty} = {cost}")

    overhead_percent = st.number_input("Overhead (%)", min_value=0.0, step=1.0)
    overhead_value = total_cost * (overhead_percent / 100)

    final_cost = total_cost + overhead_value

    st.markdown("---")
    st.subheader("สรุปต้นทุน")

    st.write("รายละเอียด:")
    for d in detail:
        st.write("-", d)

    st.write(f"ต้นทุนรวมวัสดุ: {total_cost:.2f}")
    st.write(f"Overhead: {overhead_value:.2f}")
    st.success(f"ต้นทุนรวมต่อชิ้น = {final_cost:.2f} บาท")

    if st.button("บันทึกข้อมูล"):
        st.session_state.data.append({
            "สินค้า": product,
            "เดือน": month,
            "ปี": year,
            "ต้นทุนรวม": final_cost
        })
        st.success("บันทึกข้อมูลเรียบร้อยแล้ว 🎉")

# -----------------------
# ตารางข้อมูล
# -----------------------
elif menu == "ตารางข้อมูล":
    st.title("📋 ตารางข้อมูล")

    if len(st.session_state.data) == 0:
        st.info("ยังไม่มีข้อมูล")
    else:
        df = pd.DataFrame(st.session_state.data)
        st.dataframe(df, use_container_width=True)

        st.subheader("กราฟแนวโน้ม")
        pivot = df.pivot_table(
            index=["ปี", "เดือน"],
            columns="สินค้า",
            values="ต้นทุนรวม",
            aggfunc="sum"
        )
        st.line_chart(pivot)

# -----------------------
# Export
# -----------------------
elif menu == "Export":
    st.title("📤 Export ข้อมูล")

    if len(st.session_state.data) == 0:
        st.info("ยังไม่มีข้อมูลให้ export")
    else:
        df = pd.DataFrame(st.session_state.data)

        st.download_button(
            "ดาวน์โหลดเป็น Excel",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="cost_data.csv",
            mime="text/csv"
        )
