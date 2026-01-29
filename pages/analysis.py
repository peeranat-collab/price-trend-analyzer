import streamlit as st
import pandas as pd
from datetime import datetime

# =========================
# Helper
# =========================
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
# Analysis Page
# =========================
st.title("📊 วิเคราะห์ต้นทุน (YoY Impact Analysis)")

df = load_data()

if len(df) == 0:
    st.warning("ยังไม่มีข้อมูลราคาในระบบ")
    st.stop()

# -------------------------
# 1️⃣ เลือกช่วงวิเคราะห์
# -------------------------
st.subheader("1️⃣ เลือกช่วงเวลา")

col1, col2 = st.columns(2)
with col1:
    sel_month = st.selectbox(
        "เลือกเดือน",
        list(range(1, 13)),
        format_func=lambda x: f"เดือน {x}"
    )
with col2:
    sel_year = st.selectbox(
        "เลือกปี (ปีนี้)",
        sorted(df["ปี"].unique(), reverse=True)
    )

base_year = sel_year - 1
st.caption(f"เปรียบเทียบ: {sel_month}/{sel_year} กับ {sel_month}/{base_year}")

# -------------------------
# 2️⃣ สัดส่วนต้นทุน
# -------------------------
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

for i, mat in enumerate(materials):
    with cols[i]:
        weights[mat] = st.number_input(
            mat,
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            value=0.0
        )

total_weight = sum(weights.values())
st.caption(f"รวมสัดส่วน = {total_weight:.1f}%")

if total_weight == 0:
    st.warning("กรุณาใส่สัดส่วนต้นทุน")
    st.stop()

# -------------------------
# 3️⃣ คำนวณ YoY Impact
# -------------------------
st.subheader("3️⃣ ผลการวิเคราะห์")

rows = []

for mat in materials:
    price_now = get_price(df, mat, sel_year, sel_month)
    price_prev = get_price(df, mat, base_year, sel_month)

    if price_now is None or price_prev is None:
        continue

    yoy_pct = (price_now - price_prev) / price_prev * 100
    impact = yoy_pct * (weights[mat] / total_weight)

    rows.append({
        "วัสดุ": mat,
        f"ราคา {base_year}": round(price_prev, 2),
        f"ราคา {sel_year}": round(price_now, 2),
        "YoY %": round(yoy_pct, 2),
        "สัดส่วน (%)": weights[mat],
        "Impact ต่อรวม (%)": round(impact, 2)
    })

result_df = pd.DataFrame(rows)

if len(result_df) == 0:
    st.error("ข้อมูลไม่ครบสำหรับเดือนที่เลือก")
    st.stop()

st.dataframe(result_df, use_container_width=True)

# -------------------------
# 4️⃣ Summary
# -------------------------
total_impact = result_df["Impact ต่อรวม (%)"].sum()
main_driver = result_df.sort_values(
    "Impact ต่อรวม (%)",
    ascending=False
).iloc[0]

st.markdown("---")
st.subheader("📌 สรุปผล")

if total_impact >= 0:
    st.error(f"🔺 ต้นทุนรวมเพิ่มประมาณ +{total_impact:.2f}%")
else:
    st.success(f"🔻 ต้นทุนรวมลดประมาณ {total_impact:.2f}%")

st.info(
    f"ตัวแปรหลักคือ **{main_driver['วัสดุ']}** "
    f"(Impact {main_driver['Impact ต่อรวม (%)']}%)"
)

# -------------------------
# 5️⃣ Recommendation
# -------------------------
st.subheader("💡 คำแนะนำเชิงจัดซื้อ")

if main_driver["YoY %"] > 0:
    st.write(
        f"- ควรพิจารณาล็อคราคา **{main_driver['วัสดุ']}** "
        f"เนื่องจากราคาเพิ่มขึ้น YoY {main_driver['YoY %']}%"
    )
else:
    st.write(
        f"- **{main_driver['วัสดุ']}** มีแนวโน้มลดลง "
        f"อาจชะลอการซื้อได้"
    )
