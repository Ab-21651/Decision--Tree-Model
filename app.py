import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64

# Load model
with open('best_decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Full feature list (short version for demo, use your real list)
FEATURES = [
    "SK_ID_CURR",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
    "COMMONAREA_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "FLOORSMIN_AVG",
    "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
    "APARTMENTS_MODE",
    "BASEMENTAREA_MODE",
    "YEARS_BEGINEXPLUATATION_MODE",
    "YEARS_BUILD_MODE",
    "COMMONAREA_MODE",
    "ELEVATORS_MODE",
    "ENTRANCES_MODE",
    "FLOORSMAX_MODE",
    "FLOORSMIN_MODE",
    "LANDAREA_MODE",
    "LIVINGAPARTMENTS_MODE",
    "LIVINGAREA_MODE",
    "NONLIVINGAPARTMENTS_MODE",
    "NONLIVINGAREA_MODE",
    "APARTMENTS_MEDI",
    "BASEMENTAREA_MEDI",
    "YEARS_BEGINEXPLUATATION_MEDI",
    "YEARS_BUILD_MEDI",
    "COMMONAREA_MEDI",
    "ELEVATORS_MEDI",
    "ENTRANCES_MEDI",
    "FLOORSMAX_MEDI",
    "FLOORSMIN_MEDI",
    "LANDAREA_MEDI",
    "LIVINGAPARTMENTS_MEDI",
    "LIVINGAREA_MEDI",
    "NONLIVINGAPARTMENTS_MEDI",
    "NONLIVINGAREA_MEDI",
    "TOTALAREA_MODE",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
    "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_3",
    "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_5",
    "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_7",
    "FLAG_DOCUMENT_8",
    "FLAG_DOCUMENT_9",
    "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11",
    "FLAG_DOCUMENT_12",
    "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14",
    "FLAG_DOCUMENT_15",
    "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_18",
    "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_20",
    "FLAG_DOCUMENT_21",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "AGE",
    "YEARS_EMPLOYED",
    "AGE_EMPLOYED_RATIO",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "ANNUITY_CREDIT_RATIO",
    "INCOME_PER_FAM_MEMBER",
    "REGISTRATION_AGE_DIFF",
    "PHONECHANGE_AGE_DIFF",
    "CONTACT_AVAILABILITY",
    "HAS_CAR_OLD",
    "IS_REGIONAL_MOVER",
    "SOCIAL_DEF_RATIO",
    "NUM_MISSING",
    "NAME_CONTRACT_TYPE_Revolving loans",
    "CODE_GENDER_M",
    "CODE_GENDER_XNA",
    "FLAG_OWN_CAR_Y",
    "FLAG_OWN_REALTY_Y",
    "NAME_TYPE_SUITE_Family",
    "NAME_TYPE_SUITE_Group of people",
    "NAME_TYPE_SUITE_Other_A",
    "NAME_TYPE_SUITE_Other_B",
    "NAME_TYPE_SUITE_Spouse partner",
    "NAME_TYPE_SUITE_Unaccompanied",
    "NAME_INCOME_TYPE_Commercial associate",
    "NAME_INCOME_TYPE_Maternity leave",
    "NAME_INCOME_TYPE_Pensioner",
    "NAME_INCOME_TYPE_State servant",
    "NAME_INCOME_TYPE_Student",
    "NAME_INCOME_TYPE_Unemployed",
    "NAME_INCOME_TYPE_Working",
    "NAME_EDUCATION_TYPE_Higher education",
    "NAME_EDUCATION_TYPE_Incomplete higher",
    "NAME_EDUCATION_TYPE_Lower secondary",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "NAME_FAMILY_STATUS_Married",
    "NAME_FAMILY_STATUS_Separated",
    "NAME_FAMILY_STATUS_Single / not married",
    "NAME_FAMILY_STATUS_Unknown",
    "NAME_FAMILY_STATUS_Widow",
    "NAME_HOUSING_TYPE_House / apartment",
    "NAME_HOUSING_TYPE_Municipal apartment",
    "NAME_HOUSING_TYPE_Office apartment",
    "NAME_HOUSING_TYPE_Rented apartment",
    "NAME_HOUSING_TYPE_With parents",
    "OCCUPATION_TYPE_Cleaning staff",
    "OCCUPATION_TYPE_Cooking staff",
    "OCCUPATION_TYPE_Core staff",
    "OCCUPATION_TYPE_Drivers",
    "OCCUPATION_TYPE_HR staff",
    "OCCUPATION_TYPE_High skill tech staff",
    "OCCUPATION_TYPE_IT staff",
    "OCCUPATION_TYPE_Laborers",
    "OCCUPATION_TYPE_Low-skill Laborers",
    "OCCUPATION_TYPE_Managers",
    "OCCUPATION_TYPE_Medicine staff",
    "OCCUPATION_TYPE_Private service staff",
    "OCCUPATION_TYPE_Realty agents",
    "OCCUPATION_TYPE_Sales staff",
    "OCCUPATION_TYPE_Secretaries",
    "OCCUPATION_TYPE_Security staff",
    "OCCUPATION_TYPE_Waiters/barmen staff",
    "WEEKDAY_APPR_PROCESS_START_MONDAY",
    "WEEKDAY_APPR_PROCESS_START_SATURDAY",
    "WEEKDAY_APPR_PROCESS_START_SUNDAY",
    "WEEKDAY_APPR_PROCESS_START_THURSDAY",
    "WEEKDAY_APPR_PROCESS_START_TUESDAY",
    "WEEKDAY_APPR_PROCESS_START_WEDNESDAY",
    "ORGANIZATION_TYPE_Agriculture",
    "ORGANIZATION_TYPE_Bank",
    "ORGANIZATION_TYPE_Business Entity Type 1",
    "ORGANIZATION_TYPE_Business Entity Type 2",
    "ORGANIZATION_TYPE_Business Entity Type 3",
    "ORGANIZATION_TYPE_Cleaning",
    "ORGANIZATION_TYPE_Construction",
    "ORGANIZATION_TYPE_Culture",
    "ORGANIZATION_TYPE_Electricity",
    "ORGANIZATION_TYPE_Emergency",
    "ORGANIZATION_TYPE_Government",
    "ORGANIZATION_TYPE_Hotel",
    "ORGANIZATION_TYPE_Housing",
    "ORGANIZATION_TYPE_Industry: type 1",
    "ORGANIZATION_TYPE_Industry: type 10",
    "ORGANIZATION_TYPE_Industry: type 11",
    "ORGANIZATION_TYPE_Industry: type 12",
    "ORGANIZATION_TYPE_Industry: type 13",
    "ORGANIZATION_TYPE_Industry: type 2",
    "ORGANIZATION_TYPE_Industry: type 3",
    "ORGANIZATION_TYPE_Industry: type 4",
    "ORGANIZATION_TYPE_Industry: type 5",
    "ORGANIZATION_TYPE_Industry: type 6",
    "ORGANIZATION_TYPE_Industry: type 7",
    "ORGANIZATION_TYPE_Industry: type 8",
    "ORGANIZATION_TYPE_Industry: type 9",
    "ORGANIZATION_TYPE_Insurance",
    "ORGANIZATION_TYPE_Kindergarten",
    "ORGANIZATION_TYPE_Legal Services",
    "ORGANIZATION_TYPE_Medicine",
    "ORGANIZATION_TYPE_Military",
    "ORGANIZATION_TYPE_Mobile",
    "ORGANIZATION_TYPE_Other",
    "ORGANIZATION_TYPE_Police",
    "ORGANIZATION_TYPE_Postal",
    "ORGANIZATION_TYPE_Realtor",
    "ORGANIZATION_TYPE_Religion",
    "ORGANIZATION_TYPE_Restaurant",
    "ORGANIZATION_TYPE_School",
    "ORGANIZATION_TYPE_Security",
    "ORGANIZATION_TYPE_Security Ministries",
    "ORGANIZATION_TYPE_Self-employed",
    "ORGANIZATION_TYPE_Services",
    "ORGANIZATION_TYPE_Telecom",
    "ORGANIZATION_TYPE_Trade: type 1",
    "ORGANIZATION_TYPE_Trade: type 2",
    "ORGANIZATION_TYPE_Trade: type 3",
    "ORGANIZATION_TYPE_Trade: type 4",
    "ORGANIZATION_TYPE_Trade: type 5",
    "ORGANIZATION_TYPE_Trade: type 6",
    "ORGANIZATION_TYPE_Trade: type 7",
    "ORGANIZATION_TYPE_Transport: type 1",
    "ORGANIZATION_TYPE_Transport: type 2",
    "ORGANIZATION_TYPE_Transport: type 3",
    "ORGANIZATION_TYPE_Transport: type 4",
    "ORGANIZATION_TYPE_University",
    "ORGANIZATION_TYPE_XNA",
    "FONDKAPREMONT_MODE_org spec account",
    "FONDKAPREMONT_MODE_reg oper account",
    "FONDKAPREMONT_MODE_reg oper spec account",
    "HOUSETYPE_MODE_specific housing",
    "HOUSETYPE_MODE_terraced house",
    "WALLSMATERIAL_MODE_Mixed",
    "WALLSMATERIAL_MODE_Monolithic",
    "WALLSMATERIAL_MODE_Others",
    "WALLSMATERIAL_MODE_Panel",
    "WALLSMATERIAL_MODE_Stone brick",
    "WALLSMATERIAL_MODE_Wooden",
    "EMERGENCYSTATE_MODE_Yes"
]


st.markdown("<h1 style='color: black;'>Loan Default Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='color: black;'>Predicting the likelihood of loan default based on applicant data</h3>", unsafe_allow_html=True)


# Function to encode the image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("data:image/png;base64,{encoded}");
             background-size: cover;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

add_bg_from_local('background.jpg')  # Add background image here


# User inputs
income = st.number_input("Annual Income (AMT_INCOME_TOTAL)", min_value=0)
credit = st.number_input("Credit Amount (AMT_CREDIT)", min_value=0)
annuity = st.number_input("Annuity Amount (AMT_ANNUITY)", min_value=0)
children = st.number_input("Number of Children (CNT_CHILDREN)", min_value=0)
family = st.number_input("Family Members (CNT_FAM_MEMBERS)", min_value=1)
days_birth = st.number_input("Days of Birth (negative number)", min_value=-30000, max_value=0)
days_employed = st.number_input("Days Employed (negative number)", min_value=-30000, max_value=0)
gender = st.selectbox("Gender", ["M", "F"])
own_car = st.checkbox("Owns Car?")
own_realty = st.checkbox("Owns Realty?")

if st.button("Predict"):
    # Feature engineering
    AGE = -days_birth / 365
    YEARS_EMPLOYED = -days_employed / 365
    AGE_EMPLOYED_RATIO = YEARS_EMPLOYED / AGE if AGE != 0 else 0
    CREDIT_INCOME_RATIO = credit / income if income != 0 else 0
    ANNUITY_INCOME_RATIO = annuity / income if income != 0 else 0
    ANNUITY_CREDIT_RATIO = annuity / credit if credit != 0 else 0
    INCOME_PER_FAM_MEMBER = income / family if family != 0 else 0

    # One-hot encodings
    CODE_GENDER_M = 1 if gender == "M" else 0
    FLAG_OWN_CAR_Y = 1 if own_car else 0
    FLAG_OWN_REALTY_Y = 1 if own_realty else 0

    # Build input DataFrame
    input_dict = dict.fromkeys(FEATURES, 0)
    input_dict.update({
        'AMT_INCOME_TOTAL': income,
        'AMT_CREDIT': credit,
        'AMT_ANNUITY': annuity,
        'CNT_CHILDREN': children,
        'CNT_FAM_MEMBERS': family,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'AGE': AGE,
        'YEARS_EMPLOYED': YEARS_EMPLOYED,
        'AGE_EMPLOYED_RATIO': AGE_EMPLOYED_RATIO,
        'CREDIT_INCOME_RATIO': CREDIT_INCOME_RATIO,
        'ANNUITY_INCOME_RATIO': ANNUITY_INCOME_RATIO,
        'ANNUITY_CREDIT_RATIO': ANNUITY_CREDIT_RATIO,
        'INCOME_PER_FAM_MEMBER': INCOME_PER_FAM_MEMBER,
        'CODE_GENDER_M': CODE_GENDER_M,
        'FLAG_OWN_CAR_Y': FLAG_OWN_CAR_Y,
        'FLAG_OWN_REALTY_Y': FLAG_OWN_REALTY_Y
    })

    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write(f"**Default Risk:** {'Default' if prediction == 1 else 'No Default'}")
    st.write(f"**Probability of Default:** {probability:.2%}")

