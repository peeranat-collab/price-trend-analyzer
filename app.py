import streamlit as st
import pandas as pd
from datetime import datetime
import os
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="วิเคราะห์แนวโน้มราคา", layout="wide")

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

    result["% เปลี่ยนแปลง"] = (
        (result["ปีนี้"] - result["ปีที่แล้ว"]) /
        result["ปีที่แล้ว"].replace(0, 1)
    ) * 100

    return result.reset_index()

def linear_forecast(series, periods=3):
    y = series.values.reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(y) + periods).reshape(-1, 1)
    forecast = model.predict(future_X)

    return forecast.flatten()

df_data = load_data()

st.sidebar.title("📊 เมนู")
menu = st.sidebar.radio(
    "เลือกเมนู",
    [
        "Dashboard",
        "กรอกข้อมูลต้นทุน",
        "ตารางข้อมูล",
        "วิเคราะห์แนวโน้ม",
        "คำแนะนำการจัดซื้อ",
        "พยากรณ์ราคา",
        "Export"
    ]
)

# ---------------- Dashboard ----------------
if menu == "Dashboard":
    st.title("📊 Dashboard")

    if len(df_data) == 0:
        st.info("ยังไม่มีข้อมูล")
    else:
        st.subheader("ข้อมูลล่าสุด")
        st.dataframe(df_data.tail(10), use_container_width=True)

        st.subheader("ต้นทุนรวมตามสินค้า")
        summary = df_data.groupby("สินค้า")["ต้นทุน"].sum()
        st.bar_chart(summary)

# ---------------- Input ----------------
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
        materials_base + ["วัสดุอื่นๆ"]
    )

    overhead_percent = st.number_input("Overhead (%)", min_value=0.0, step=1.0)

    material_rows = []

    st.markdown("---")

    for mat in selected_materials:
        st.markdown(f"### {mat}")
        c1, c2 = st.columns(2)

        with c1:
            price = st.number_input(
                f"ราคา/หน่วย ({mat})",
                min_value=0.0,
                step=1.0,
                key=f"p_{mat}"
            )

        with c2:
            qty = st.number_input(
                f"ปริมาณที่ใช้ ({mat})",
                min_value=0.0,
                step=0.1,
                key=f"q_{mat}"
            )

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

# ---------------- Table ----------------
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

# ---------------- Trend Analysis ----------------
elif menu == "วิเคราะห์แนวโน้ม":
    st.title("📈 วิเคราะห์แนวโน้มวัสดุ (YoY)")

    if len(df_data) == 0:
        st.info("ยังไม่มีข้อมูล")
    else:
        col1, col2 = st.columns(2)
        with col1:
            sel_month = st.selectbox("เลือกเดือน", sorted(df_data["เดือน"].unique()))
        with col2:
            sel_year = st.selectbox("เลือกปี", sorted(df_data["ปี"].unique()))

        result = yoy_compare(df_data, sel_month, sel_year)

        if len(result) == 0:
            st.warning("ไม่มีข้อมูลปีที่แล้วสำหรับเปรียบเทียบ")
        else:
            st.subheader("ตารางเปรียบเทียบ YoY")
            st.dataframe(result, use_container_width=True)

            st.subheader("กราฟเปรียบเทียบ")
            chart_df = result.set_index("วัสดุ")[["ปีที่แล้ว", "ปีนี้"]]
            st.bar_chart(chart_df)

            st.subheader("สรุปแนวโน้ม (ภาษาไทย)")
            for _, row in result.iterrows():
                mat = row["วัสดุ"]
                pct = row["% เปลี่ยนแปลง"]

                if pct > 0:
                    st.write(f"- {mat}: ↑ เพิ่มขึ้น {pct:.2f}%")
                elif pct < 0:
                    st.write(f"- {mat}: ↓ ลดลง {abs(pct):.2f}%")
                else:
                    st.write(f"- {mat}: คงที่")

# ---------------- Recommendation Engine ----------------
elif menu == "คำแนะนำการจัดซื้อ":
    st.title("💡 คำแนะนำการจัดซื้อ (งวดล่าสุด)")

    if len(df_data) == 0:
        st.info("ยังไม่มีข้อมูล")
    else:
        latest_year = df_data["ปี"].max()
        latest_month = df_data[df_data["ปี"] == latest_year]["เดือน"].max()

        st.write(f"📌 ใช้ข้อมูลงวดล่าสุด: {latest_month}/{latest_year}")

        current_data = df_data[
            (df_data["ปี"] == latest_year) &
            (df_data["เดือน"] == latest_month)
        ]

        total_cost_now = current_data["ต้นทุน"].sum()

        yoy_result = yoy_compare(df_data, latest_month, latest_year)

        st.subheader("แนวโน้มวัสดุ (YoY)")
        st.dataframe(yoy_result, use_container_width=True)

        avg_change = yoy_result["% เปลี่ยนแปลง"].mean()
        recommended_price = total_cost_now * (1 + avg_change / 100)

        st.markdown("---")
        st.subheader("📌 สรุปคำแนะนำ")

        st.write(f"ต้นทุนปัจจุบัน: {total_cost_now:,.2f} บาท")

        if avg_change > 0:
            st.write(f"แนวโน้มเฉลี่ย: เพิ่มขึ้น {avg_change:.2f}%")
        else:
            st.write(f"แนวโน้มเฉลี่ย: ลดลง {abs(avg_change):.2f}%")

        st.success(f"👉 ควรซื้อไม่เกิน: {recommended_price:,.2f} บาท")

        st.subheader("เหตุผล")
        for _, row in yoy_result.iterrows():
            mat = row["วัสดุ"]
            pct = row["% เปลี่ยนแปลง"]
            if pct > 0:
                st.write(f"- {mat} เพิ่มขึ้น {pct:.2f}%")
            elif pct < 0:
                st.write(f"- {mat} ลดลง {abs(pct):.2f}%")

# ---------------- Forecast ----------------
elif menu == "พยากรณ์ราคา":
    st.title("🔮 พยากรณ์ราคา (Linear Regression)")

    if len(df_data) == 0:
        st.info("ยังไม่มีข้อมูล")
    else:
        material = st.selectbox("เลือกวัสดุ", sorted(df_data["วัสดุ"].unique()))
        periods = st.selectbox("พยากรณ์ล่วงหน้า (เดือน)", [3, 6, 12])

        mat_df = df_data[df_data["วัสดุ"] == material]
        mat_df = mat_df.groupby(["ปี", "เดือน"])["ต้นทุน"].sum().reset_index()
        mat_df["time_index"] = range(len(mat_df))

        if len(mat_df) < 3:
            st.warning("ข้อมูลน้อยเกินไปสำหรับการพยากรณ์")
        else:
            forecast_values = linear_forecast(mat_df["ต้นทุน"], periods)

            hist = forecast_values[:len(mat_df)]
            future = forecast_values[len(mat_df):]

            hist_df = pd.DataFrame({
                "งวด": mat_df["time_index"],
                "ต้นทุน": hist
            })

            future_df = pd.DataFrame({
                "งวด": range(len(mat_df), len(mat_df) + periods),
                "ต้นทุน": future
            })

            st.subheader("กราฟย้อนหลัง + พยากรณ์")
            chart_df = pd.concat([hist_df, future_df])
            chart_df = chart_df.set_index("งวด")

            st.line_chart(chart_df)

            change_pct = ((future[-1] - hist[-1]) / hist[-1]) * 100

            st.subheader("สรุปการพยากรณ์")
            if change_pct > 0:
                st.write(f"คาดว่าราคาจะเพิ่มขึ้นประมาณ {change_pct:.2f}% ใน {periods} เดือน")
            else:
                st.write(f"คาดว่าราคาจะลดลงประมาณ {abs(change_pct):.2f}% ใน {periods} เดือน")

# ---------------- Export ----------------
elif menu == "Export":
    st.title("📤 Export ข้อมูล")

    if len(df_data) == 0:
        st.info("ยังไม่มีข้อมูลให้ export")
    else:
        st.download_button(
            "ดาวน์โหลดเป็น CSV (เปิดใน Excel ได้)",
            data=df_data.to_csv(index=False).encode("utf-8-sig"),
            file_name="cost_data.csv",
            mime="text/csv"
        )
