import pandas as pd

REQUIRED_COLUMNS = ["ประเภท", "ราคา", "วันที่เริ่ม", "วันที่สิ้นสุด", "สัปดาห์"]

def load_pet_excel(file):
    try:
        df = pd.read_excel(file)

        # ตรวจสอบ column
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return {
                "status": "error",
                "message": f"ไฟล์ขาด column: {missing}"
            }

        # กรองเฉพาะ PET
        pet_df = df[df["ประเภท"].astype(str).str.upper().str.contains("PET")].copy()

        if pet_df.empty:
            return {
                "status": "error",
                "message": "ไม่พบข้อมูล PET ในไฟล์"
            }

        # แปลงวันที่
        pet_df["วันที่เริ่ม"] = pd.to_datetime(pet_df["วันที่เริ่ม"])
        pet_df["วันที่สิ้นสุด"] = pd.to_datetime(pet_df["วันที่สิ้นสุด"])

        return {
            "status": "success",
            "data": pet_df
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
