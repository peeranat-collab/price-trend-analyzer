import streamlit as st
import pandas as pd
from datetime import datetime
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# PDF & Plot
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
import matplotlib.pyplot as plt

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

# ---------------- Utilities ----------------
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

def thai_date(d: datetime):
    months = [
        "มกราคม","กุมภาพันธ์","มีนาคม","เมษายน","พฤษภาคม","มิถุนายน",
        "กรกฎาคม","สิงหาคม","กันยายน","ตุลาคม","พฤศจิกายน","ธันวาคม"
    ]
    return f"{d.day} {months[d.month-1]} {d.year}"

# ---------------- PDF Helpers ----------------
def save_trend_plot(df, filename):
    plt.figure()
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_bar_plot(df, filename):
    plt.figure()
    df.plot(kind="bar")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def build_pdf(
    filepath,
    header_th, header_en, department,
    report_title_th, report_title_en,
    summary_th, summary_en,
    tables_and_images
):
    styles = getSampleStyleSheet()
    story = []

    if header_th or header_en or department:
        if header_th:
            story.append(Paragraph(header_th, styles["Title"]))
        if header_en:
            story.append(Paragraph(header_en, styles["Normal"]))
        if department:
            story.append(Paragraph(department, styles["Normal"]))
        story.append(Spacer(1, 1*cm))

    story.append(Paragraph(report_title_en, styles["Heading1"]))
    story.append(Paragraph(report_title_th, styles["Heading2"]))
    story.append(Spacer(1, 1*cm))

    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    story.append(Paragraph(summary_en, styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("สรุปผู้บริหาร", styles["Heading2"]))
    story.append(Paragraph(summary_th, styles["Normal"]))
    story.append(PageBreak())

    for item in tables_and_images:
        if item["type"] == "table":
            story.append(Paragraph(item["title"], styles["Heading2"]))
            story.append(Spacer(1, 0.3*cm))
            story.append(item["content"])
            story.append(PageBreak())
        elif item["type"] == "image":
            story.append(Paragraph(item["title"], styles["Heading2"]))
            story.append(Spacer(1, 0.3*cm))
            story.append(Image(item["content"], width=16*cm, height=9*cm))
            story.append(PageBreak())

    doc = SimpleDocTemplate(filepath, pagesize=A4)
    doc.build(story)

# ---------------- Load ----------------
df_data = load_data()

# ---------------- Sidebar ----------------
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
        "รายงาน PDF",
        "Export"
    ]
)

# -------- Dashboard --------
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

# -------- Input --------
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
                min_value=0.0, step=1.0, key=f"p_{mat}"
            )
        with c2:
            qty = st.number_input(
                f"ปริมาณที่ใช้ ({mat})",
                min_value=0.0, step=0.1, key=f"q_{mat}"
            )
        cost = price * qty
        material_rows.append({
            "สินค้า": product, "เดือน": month, "ปี": year, "วัสดุ": mat,
            "ราคา/หน่วย": price, "ปริมาณ": qty, "ต้นทุน": cost,
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
        st.write(f"ต้นทุนรวมวัสดุ: {base_total:,.2f} บาท")
        st.write(f"Overhead: {overhead_value:,.2f} บาท")
        st.success(f"ต้นทุนรวมต่อสินค้า = {final_total:,.2f} บาท")

        if st.button("บันทึกข้อมูล"):
            new_df = pd.DataFrame(material_rows)
            df_all = pd.concat([df_data, new_df], ignore_index=True)
            save_data(df_all)
            st.success("บันทึกข้อมูลเรียบร้อยแล้ว 🎉")
            st.experimental_rerun()

# -------- Table --------
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

# -------- Trend --------
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
            st.dataframe(result, use_container_width=True)
            st.bar_chart(result.set_index("วัสดุ")[["ปีที่แล้ว", "ปีนี้"]])

# -------- Recommendation --------
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
        avg_change = yoy_result["% เปลี่ยนแปลง"].mean()
        recommended_price = total_cost_now * (1 + avg_change / 100)

        st.subheader("แนวโน้มวัสดุ (YoY)")
        st.dataframe(yoy_result, use_container_width=True)

        st.markdown("---")
        st.write(f"ต้นทุนปัจจุบัน: {total_cost_now:,.2f} บาท")
        if avg_change > 0:
            st.write(f"แนวโน้มเฉลี่ย: เพิ่มขึ้น {avg_change:.2f}%")
        else:
            st.write(f"แนวโน้มเฉลี่ย: ลดลง {abs(avg_change):.2f}%")

        st.success(f"👉 ควรซื้อไม่เกิน: {recommended_price:,.2f} บาท")

# -------- Forecast --------
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

            hist_df = pd.DataFrame({"งวด": mat_df["time_index"], "ต้นทุน": hist})
            future_df = pd.DataFrame({
                "งวด": range(len(mat_df), len(mat_df) + periods),
                "ต้นทุน": future
            })

            chart_df = pd.concat([hist_df, future_df]).set_index("งวด")
            st.line_chart(chart_df)

# -------- PDF Report --------
elif menu == "รายงาน PDF":
    st.title("📄 สร้างรายงาน PDF")

    header_th = st.text_input("ชื่อบริษัท (TH) – ใส่หรือเว้นว่างได้")
    header_en = st.text_input("Company Name (EN)")
    department = st.text_input("ชื่อแผนก / Department")

    report_title_th = "รายงานวิเคราะห์ต้นทุนและพยากรณ์ราคา"
    report_title_en = "Cost Analysis & Forecast Report"

    summary_th = "รายงานฉบับนี้จัดทำขึ้นเพื่อสรุปแนวโน้มต้นทุน วิเคราะห์ YoY และคาดการณ์ราคาในอนาคต"
    summary_en = "This report summarizes cost trends, YoY analysis, and future forecasts."

    if st.button("📥 สร้าง PDF"):
        filepath = "cost_report.pdf"

        tables_and_images = []
        if len(df_data) > 0:
            tbl_data = [df_data.columns.tolist()] + df_data.head(20).values.tolist()
            table = Table(tbl_data)
            table.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)
            ]))
            tables_and_images.append({
                "type": "table",
                "title": "ข้อมูลตัวอย่าง",
                "content": table
            })

        build_pdf(
            filepath,
            header_th, header_en, department,
            report_title_th, report_title_en,
            summary_th, summary_en,
            tables_and_images
        )

        with open(filepath, "rb") as f:
            st.download_button(
                "⬇️ ดาวน์โหลด PDF",
                f,
                file_name="Cost_Report.pdf",
                mime="application/pdf"
            )

# -------- Export --------
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
