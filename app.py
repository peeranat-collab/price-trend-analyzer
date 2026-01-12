import streamlit as st
import pandas as pd
from datetime import datetime
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scrapers.bangchak_diesel import get_monthly_average


# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors

# Plot
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cost Intelligence System", layout="wide")

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
# Utilities
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
        "Last Year": prev_sum,
        "This Year": cur_sum
    }).fillna(0)

    result["Change %"] = (
        (result["This Year"] - result["Last Year"]) /
        result["Last Year"].replace(0, 1)
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

# =========================
# Corporate Plot Export
# =========================
def save_trend_plot(df, filename, title):
    plt.figure(figsize=(8,4))
    for col in df.columns:
        plt.plot(df.index, df[col], marker="o", label=col)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_bar_plot(df, filename, title):
    plt.figure(figsize=(8,4))
    df.plot(kind="bar")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# =========================
# Corporate PDF Builder
# =========================
def build_corporate_pdf(
    filepath,
    header_th,
    header_en,
    department,
    report_title_th,
    report_title_en,
    exec_summary_th,
    exec_summary_en,
    sections
):
    styles = getSampleStyleSheet()

    styles["Title"].alignment = TA_CENTER
    styles["Heading1"].alignment = TA_LEFT

    cover_title = ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=20
    )

    subtitle = ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=12,
        textColor=colors.grey
    )

    h1 = ParagraphStyle(
        name="H1",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=12
    )

    normal = styles["Normal"]

    story = []

    # -------- Cover --------
    if header_th:
        story.append(Paragraph(header_th, cover_title))
    if header_en:
        story.append(Paragraph(header_en, subtitle))
    if department:
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph(department, subtitle))

    story.append(Spacer(1, 2*cm))
    story.append(Paragraph(report_title_en, cover_title))
    story.append(Paragraph(report_title_th, subtitle))

    today = thai_date(datetime.today())
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph(f"Generated on: {today}", subtitle))
    story.append(PageBreak())

    # -------- Executive Summary --------
    story.append(Paragraph("Executive Summary", h1))
    story.append(Paragraph(exec_summary_en, normal))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("สรุปผู้บริหาร", h1))
    story.append(Paragraph(exec_summary_th, normal))
    story.append(PageBreak())

    # -------- Sections --------
    for sec in sections:
        story.append(Paragraph(sec["title"], h1))
        story.append(Spacer(1, 0.3*cm))

        if sec["type"] == "table":
            story.append(sec["content"])
        elif sec["type"] == "image":
            story.append(Image(sec["content"], width=16*cm, height=9*cm))
        elif sec["type"] == "text":
            story.append(Paragraph(sec["content"], normal))

        story.append(PageBreak())

    doc = SimpleDocTemplate(filepath, pagesize=A4)
    doc.build(story)

# =========================
# Load
# =========================
df_data = load_data()

# =========================
# Sidebar
# =========================
st.sidebar.title("📊 Cost Intelligence")
menu = st.sidebar.radio(
    "เลือกเมนู",
    [
        "Dashboard",
        "กรอกข้อมูลต้นทุน",
        "ตารางข้อมูล",
        "วิเคราะห์แนวโน้ม",
        "คำแนะนำการจัดซื้อ",
        "พยากรณ์ราคา",
        "รายงาน PDF (Corporate)",
        "🔄 อัปเดตราคาน้ำมัน (ดีเซล)",
        "Export"
    ]
)

# =========================
# Dashboard
# =========================
if menu == "Dashboard":
    st.title("📊 Dashboard")
    if len(df_data) == 0:
        st.info("ยังไม่มีข้อมูล")
    else:
        st.subheader("Latest Records")
        st.dataframe(df_data.tail(10), use_container_width=True)

        st.subheader("Total Cost by Product")
        summary = df_data.groupby("สินค้า")["ต้นทุน"].sum()
        st.bar_chart(summary)

# =========================
# Input
# =========================
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
        st.write(f"ต้นทุนรวมวัสดุ: {base_total:,.2f} บาท")
        st.write(f"Overhead: {overhead_value:,.2f} บาท")
        st.success(f"ต้นทุนรวมต่อสินค้า = {final_total:,.2f} บาท")

        if st.button("บันทึกข้อมูล"):
            new_df = pd.DataFrame(material_rows)
            df_all = pd.concat([df_data, new_df], ignore_index=True)
            save_data(df_all)
            st.success("บันทึกข้อมูลเรียบร้อยแล้ว 🎉")
            st.experimental_rerun()

# =========================
# Table
# =========================
elif menu == "ตารางข้อมูล":
    st.title("📋 ตารางข้อมูล")
    if len(df_data) == 0:
        st.info("ยังไม่มีข้อมูล")
    else:
        st.dataframe(df_data, use_container_width=True)

# =========================
# Trend
# =========================
elif menu == "วิเคราะห์แนวโน้ม":
    st.title("📈 วิเคราะห์แนวโน้ม (YoY)")
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
            st.bar_chart(result.set_index("วัสดุ")[["Last Year", "This Year"]])

# =========================
# Recommendation
# =========================
elif menu == "คำแนะนำการจัดซื้อ":
    st.title("💡 Recommendation")
    if len(df_data) == 0:
        st.info("ยังไม่มีข้อมูล")
    else:
        latest_year = df_data["ปี"].max()
        latest_month = df_data[df_data["ปี"] == latest_year]["เดือน"].max()

        current_data = df_data[
            (df_data["ปี"] == latest_year) &
            (df_data["เดือน"] == latest_month)
        ]

        total_cost_now = current_data["ต้นทุน"].sum()
        yoy_result = yoy_compare(df_data, latest_month, latest_year)

        avg_change = yoy_result["Change %"].mean()
        recommended_price = total_cost_now * (1 + avg_change / 100)

        st.subheader("วัสดุที่มีผลต่อราคา")
        st.dataframe(yoy_result, use_container_width=True)

        st.markdown("---")
        st.write(f"ต้นทุนปัจจุบัน: {total_cost_now:,.2f} บาท")
        st.write(f"แนวโน้มเฉลี่ย: {avg_change:.2f}%")
        st.success(f"👉 ควรซื้อไม่เกิน: {recommended_price:,.2f} บาท")

# =========================
# Forecast
# =========================
elif menu == "พยากรณ์ราคา":
    st.title("🔮 Forecast")
    if len(df_data) == 0:
        st.info("ยังไม่มีข้อมูล")
    else:
        material = st.selectbox("เลือกวัสดุ", sorted(df_data["วัสดุ"].unique()))
        periods = st.selectbox("พยากรณ์ล่วงหน้า (เดือน)", [3, 6, 12])

        mat_df = df_data[df_data["วัสดุ"] == material]
        mat_df = mat_df.groupby(["ปี", "เดือน"])["ต้นทุน"].sum().reset_index()
        mat_df["t"] = range(len(mat_df))

        if len(mat_df) < 3:
            st.warning("ข้อมูลไม่พอสำหรับพยากรณ์")
        else:
            forecast_values = linear_forecast(mat_df["ต้นทุน"], periods)

            hist = forecast_values[:len(mat_df)]
            future = forecast_values[len(mat_df):]

            chart_df = pd.DataFrame({
                "Index": list(range(len(hist))) + list(range(len(hist), len(hist) + len(future))),
                "Cost": list(hist) + list(future)
            }).set_index("Index")

            st.line_chart(chart_df)

# =========================
# Corporate PDF
# =========================
elif menu == "รายงาน PDF (Corporate)":
    st.title("📄 Corporate PDF Report")

    header_th = st.text_input("ชื่อบริษัท (TH)")
    header_en = st.text_input("Company Name (EN)")
    department = st.text_input("Department")

    report_title_th = "รายงานวิเคราะห์ต้นทุนและพยากรณ์ราคา"
    report_title_en = "Cost Analysis & Forecast Report"

    if st.button("📥 Generate PDF"):
        filepath = "Corporate_Report.pdf"

        exec_summary_th = "รายงานฉบับนี้สรุปแนวโน้มต้นทุน วิเคราะห์ YoY และคาดการณ์ราคาในอนาคต"
        exec_summary_en = "This report summarizes cost trends, YoY analysis, and future forecasts."

        sections = []

        if len(df_data) > 0:
            tbl_data = [df_data.columns.tolist()] + df_data.head(20).values.tolist()
            table = Table(tbl_data)
            table.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)
            ]))

            sections.append({
                "title": "Sample Data",
                "type": "table",
                "content": table
            })

        build_corporate_pdf(
            filepath,
            header_th,
            header_en,
            department,
            report_title_th,
            report_title_en,
            exec_summary_th,
            exec_summary_en,
            sections
        )

        with open(filepath, "rb") as f:
            st.download_button(
                "⬇️ Download PDF",
                f,
                file_name="Corporate_Report.pdf",
                mime="application/pdf"
            )

# =========================
# Export
# =========================
elif menu == "Export":
    st.title("📤 Export Data")
    if len(df_data) == 0:
        st.info("ยังไม่มีข้อมูล")
    else:
        st.download_button(
            "Download CSV",
            data=df_data.to_csv(index=False).encode("utf-8-sig"),
            file_name="cost_data.csv",
            mime="text/csv"
        )
elif menu == "🔄 อัปเดตราคาน้ำมัน (ดีเซล)":
    st.title("🔄 อัปเดตราคาน้ำมันดีเซล (Bangchak)")

    log_file = "auto/auto_log.txt"

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            logs = f.readlines()[-5:]

        st.subheader("📜 สถานะ Auto ล่าสุด")
        for l in logs:
            if "FAILED" in l:
                st.error(l.strip())
            elif "SUCCESS" in l:
                st.success(l.strip())
            else:
                st.info(l.strip())


        st.info("ระบบจะดึงราคาดีเซลจาก Bangchak และคำนวณค่าเฉลี่ยรายเดือน")

        col1, col2 = st.columns(2)
        with col1:
            sel_month = st.selectbox("เลือกเดือน", list(range(1, 13)))
        with col2:
            sel_year = st.selectbox("เลือกปี", list(range(2020, 2035)))

        if st.button("ดึงข้อมูลจาก Bangchak"):
            result = get_monthly_average(sel_year, sel_month)

            st.session_state["diesel_fetch_result"] = result
            st.session_state["diesel_month"] = sel_month
            st.session_state["diesel_year"] = sel_year

        if "diesel_fetch_result" in st.session_state:
            result = st.session_state["diesel_fetch_result"]

            if isinstance(result, dict) and result.get("status") == "fallback":
                st.warning("⚠ ไม่สามารถดึงข้อมูลอัตโนมัติได้")
                st.write("เหตุผล:", result.get("reason"))

                st.subheader("กรอกราคาดีเซลเอง (Fallback Mode)")
                manual_price = st.number_input("ราคาดีเซลเฉลี่ย (บาท/ลิตร)", min_value=0.0, step=0.1)

                st.session_state["diesel_manual_price"] = manual_price

            else:
                st.success("✅ ดึงข้อมูลสำเร็จ")
                st.write(f"ค่าเฉลี่ยราคาดีเซล = {result} บาท/ลิตร")

                st.session_state["diesel_auto_price"] = result
                st.markdown("---")

        # ปุ่มบันทึกข้อมูล
        if "diesel_fetch_result" in st.session_state:
            if st.button("💾 บันทึกเข้าระบบ"):
                month = st.session_state.get("diesel_month")
                year = st.session_state.get("diesel_year")

                # เลือกราคาอัตโนมัติ หรือ fallback
                if "diesel_auto_price" in st.session_state:
                    final_price = st.session_state["diesel_auto_price"]
                else:
                    final_price = st.session_state.get("diesel_manual_price")

                if final_price is None or final_price <= 0:
                    st.error("กรุณาระบุราคาดีเซลที่ถูกต้อง")
                else:
                    new_rows = []

                    for product in products:  # ผูกกับทุกสินค้า
                        new_rows.append({
                            "สินค้า": product,
                            "เดือน": month,
                            "ปี": year,
                            "วัสดุ": "ค่าขนส่ง (น้ำมันดีเซล)",
                            "ราคา/หน่วย": final_price,
                            "ปริมาณ": 1,
                            "ต้นทุน": final_price,
                            "overhead_percent": 0,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                    new_df = pd.DataFrame(new_rows)

                    # โหลดข้อมูลเดิม
                    old_df = load_data()

                    # ลบข้อมูลซ้ำ (เดือน/ปี/วัสดุ/สินค้าเดียวกัน)
                    if len(old_df) > 0:
                        old_df = old_df[
                            ~(
                                (old_df["วัสดุ"] == "ค่าขนส่ง (น้ำมันดีเซล)") &
                                (old_df["เดือน"] == month) &
                                (old_df["ปี"] == year)
                            )
                        ]

                    final_df = pd.concat([old_df, new_df], ignore_index=True)
                    save_data(final_df)

                    # ล้าง session
                    for k in ["diesel_fetch_result", "diesel_auto_price", "diesel_manual_price"]:
                        if k in st.session_state:
                            del st.session_state[k]

                    st.success("บันทึกราคาน้ำมันดีเซลเข้าระบบเรียบร้อยแล้ว 🎉")
                    st.experimental_rerun()


