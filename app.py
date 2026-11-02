from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "churn_pipeline.joblib"

# The columns the model was trained on, in the same order as the notebook.
FEATURES = [
    "Tenure", "PreferredLoginDevice", "CityTier", "WarehouseToHome", "PreferredPaymentMode",
    "Gender", "HourSpendOnApp", "NumberOfDeviceRegistered", "PreferedOrderCat",
    "SatisfactionScore", "MaritalStatus", "NumberOfAddress", "Complain",
    "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount", "DaySinceLastOrder",
    "CashbackAmount",
]

# Made-up example customers, only to save typing. They are not real customers.
EXAMPLES = {
    "Typical customer": dict(Tenure=9, WarehouseToHome=14, HourSpendOnApp=3, NumberOfDeviceRegistered=4,
                             SatisfactionScore=3, NumberOfAddress=3, Complain=0, OrderAmountHikeFromlastYear=15,
                             CouponUsed=1, OrderCount=2, DaySinceLastOrder=3, CashbackAmount=163),
    "New customer, complained, 6 addresses": dict(Tenure=1, WarehouseToHome=20, HourSpendOnApp=2, NumberOfDeviceRegistered=3,
                                        SatisfactionScore=2, NumberOfAddress=6, Complain=1, OrderAmountHikeFromlastYear=13,
                                        CouponUsed=0, OrderCount=1, DaySinceLastOrder=1, CashbackAmount=120),
    "Long-time customer, no complaint": dict(Tenure=30, WarehouseToHome=10, HourSpendOnApp=3, NumberOfDeviceRegistered=4,
                                             SatisfactionScore=4, NumberOfAddress=4, Complain=0, OrderAmountHikeFromlastYear=16,
                                             CouponUsed=2, OrderCount=3, DaySinceLastOrder=4, CashbackAmount=190),
}


def clean(d):
    # Same label fixes as in the notebook (Step 3).
    d = d.copy()
    d["PreferredLoginDevice"] = d["PreferredLoginDevice"].replace({"Phone": "Mobile Phone"})
    d["PreferredPaymentMode"] = d["PreferredPaymentMode"].replace({"CC": "Credit Card", "COD": "Cash on Delivery"})
    return d


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def score_many(model, df):
    # Score every row, then flag the 10% with the highest risk.
    df = df.copy()
    X = clean(df[FEATURES])
    df["churn_risk"] = model.predict_proba(X)[:, 1].round(3)
    df["rank"] = df["churn_risk"].rank(ascending=False, method="first").astype(int)
    n_contact = max(1, int(len(df) * 0.10))
    df["contact_first"] = df["rank"] <= n_contact
    first = [c for c in ["CustomerID"] if c in df.columns] + ["churn_risk", "rank", "contact_first"]
    others = [c for c in df.columns if c not in first]
    return df[first + others].sort_values("rank"), n_contact


def read_upload(file):
    if file.name.lower().endswith((".xlsx", ".xls")):
        sheets = pd.ExcelFile(file).sheet_names
        return pd.read_excel(file, sheet_name="E Comm" if "E Comm" in sheets else sheets[0])
    return pd.read_csv(file)


st.set_page_config(page_title="Customer churn demo", layout="centered")
st.title("E-commerce customer churn demo")
st.caption("Trained on a public Kaggle dataset. This is a portfolio demo, not a tool for real business decisions.")

model = load_model()
tab_one, tab_many = st.tabs(["One customer", "Many customers"])

with tab_one:
    choice = st.selectbox("Start from an example", list(EXAMPLES))
    e = EXAMPLES[choice]
    left, right = st.columns(2)

    with left:
        tenure = st.number_input("Tenure (months)", 0, 61, e["Tenure"])
        device = st.selectbox("Preferred login device", ["Mobile Phone", "Computer"])
        city = st.selectbox("City tier", [1, 2, 3])
        warehouse = st.number_input("Distance warehouse to home", 5, 127, e["WarehouseToHome"])
        payment = st.selectbox("Preferred payment mode",
                               ["Debit Card", "Credit Card", "E wallet", "UPI", "Cash on Delivery"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        hours = st.number_input("Hours on app", 0, 5, e["HourSpendOnApp"])
        devices = st.number_input("Devices registered", 1, 6, e["NumberOfDeviceRegistered"])
        category = st.selectbox("Preferred order category",
                                ["Laptop & Accessory", "Mobile Phone", "Mobile", "Fashion", "Grocery", "Others"])

    with right:
        satisfaction = st.slider("Satisfaction score", 1, 5, e["SatisfactionScore"])
        marital = st.selectbox("Marital status", ["Married", "Single", "Divorced"])
        addresses = st.number_input("Number of addresses", 1, 22, e["NumberOfAddress"])
        complain = st.selectbox("Complained last month?", [0, 1], index=e["Complain"],
                                format_func=lambda v: "Yes" if v else "No")
        hike = st.number_input("Order amount hike from last year (%)", 11, 26, e["OrderAmountHikeFromlastYear"])
        coupons = st.number_input("Coupons used", 0, 16, e["CouponUsed"])
        orders = st.number_input("Order count", 1, 16, e["OrderCount"])
        days = st.number_input("Days since last order", 0, 46, e["DaySinceLastOrder"])
        cashback = st.number_input("Cashback amount", 0, 325, e["CashbackAmount"])

    row = dict(Tenure=tenure, PreferredLoginDevice=device, CityTier=city, WarehouseToHome=warehouse,
               PreferredPaymentMode=payment, Gender=gender, HourSpendOnApp=hours,
               NumberOfDeviceRegistered=devices, PreferedOrderCat=category, SatisfactionScore=satisfaction,
               MaritalStatus=marital, NumberOfAddress=addresses, Complain=complain,
               OrderAmountHikeFromlastYear=hike, CouponUsed=coupons, OrderCount=orders,
               DaySinceLastOrder=days, CashbackAmount=cashback)

    risk = model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1]
    st.metric("Churn risk score", f"{risk:.0%}")
    st.progress(float(risk))
    st.caption("Use this as a ranking score. I haven't checked whether a score of 70% really means "
               "70 out of 100 such customers leave.")

with tab_many:
    st.write("Upload a CSV or Excel file with one row per customer. The Kaggle file works as it is. "
             "The app scores everyone and flags the 10% with the highest risk.")
    template = pd.DataFrame([{**EXAMPLES["Typical customer"], "PreferredLoginDevice": "Mobile Phone", "CityTier": 1,
                              "PreferredPaymentMode": "Debit Card", "Gender": "Male",
                              "PreferedOrderCat": "Mobile Phone", "MaritalStatus": "Single"}])[FEATURES]
    st.download_button("Download a template (one made-up row)", template.to_csv(index=False), "template.csv")

    upload = st.file_uploader("Customer file", type=["csv", "xlsx", "xls"])
    if upload is not None:
        data = read_upload(upload)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error("These columns are missing: " + ", ".join(missing))
        else:
            result, n_contact = score_many(model, data)
            st.success(f"Scored {len(result)} customers. The top {n_contact} are flagged as 'contact first'.")
            st.dataframe(result.head(200), hide_index=True)
            st.download_button("Download results", result.to_csv(index=False), "scored_customers.csv")
