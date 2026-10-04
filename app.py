import streamlit as st
import pandas as pd
import numpy as np
import portable


st.set_page_config(
    page_title="Study Office",
    layout="wide"
)

st.title("Study Office")
st.write("Students who may need support after week 6")


URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"

history = pd.read_csv(URL + "history_week6.csv")
new = pd.read_csv(URL + "new_week6.csv")

val = history[history["cohort"] == 2025].copy()


model = portable.Model("model")

val["risk"] = model.predict_proba(val)
new["risk"] = model.predict_proba(new)


st.header("This week's list")

new["rank"] = new["risk"].rank(
    ascending=False,
    method="first"
).astype(int)

new["Talk to this student"] = new["rank"] <= 40

students = new.sort_values("risk", ascending=False)

st.write(
    "The 40 students with the highest predicted risk are marked for contact."
)

st.dataframe(
    students,
    use_container_width=True,
    hide_index=True
)


st.header("The mistakes of a rule")

with st.sidebar:
    st.header("Cost assumptions")

    COST_TALK = st.number_input(
        "Cost of one conversation in DKK",
        min_value=0,
        value=500,
        step=100
    )

    COST_WORRY = st.number_input(
        "Cost of worrying a student unnecessarily in DKK",
        min_value=0,
        value=2000,
        step=500
    )

    COST_LEAVE = st.number_input(
        "Cost when a student leaves in DKK",
        min_value=0,
        value=60000,
        step=5000
    )

    HELPS = st.slider(
        "Share of contacted students at risk who stay because of the conversation",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05
    )

    st.caption(
        "These are assumptions and can be changed to see how the preferred number of conversations changes."
    )


number = st.slider(
    "Number of students to contact",
    min_value=1,
    max_value=min(200, len(val)),
    value=40
)

val["contacted"] = (
    val["risk"].rank(
        ascending=False,
        method="first"
    ) <= number
)


TP = ((val["left"] == 1) & (val["contacted"] == True)).sum()
FP = ((val["left"] == 0) & (val["contacted"] == True)).sum()
FN = ((val["left"] == 1) & (val["contacted"] == False)).sum()
TN = ((val["left"] == 0) & (val["contacted"] == False)).sum()


precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0

Talked = COST_TALK * (TP + FP)
Worried = COST_WORRY * FP
Left = (
    COST_LEAVE * FN
    + COST_LEAVE * TP * (1 - HELPS)
)
total_cost = Talked + Worried + Left

st.markdown(
    f"""
    <div style="
        position: fixed;
        top: 5rem;
        right: 2rem;
        z-index: 9999;
        background: white;
        color: #111;
        border: 1px solid #d9d9d9;
        border-radius: 12px;
        padding: 14px 18px;
        min-width: 220px;
        text-align: center;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.14);
    ">
        <div style="font-size: 0.85rem; margin-bottom: 4px;">
            Estimated total cost
        </div>
        <div style="font-size: 1.7rem; font-weight: 700;">
            {total_cost:,.0f} DKK
        </div>
        <div style="font-size: 0.8rem; margin-top: 4px;">
            {number} students contacted
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Reached in time",
    TP
)

c2.metric(
    "Worried for nothing",
    FP
)

c3.metric(
    "Missed",
    FN
)

c4.metric(
    "Correctly left alone",
    TN
)


c1, c2 = st.columns(2)

c1.metric(
    "Precision",
    f"{precision:.1%}"
)

c2.metric(
    "Recall",
    f"{recall:.1%}"
)


st.write(
    f"If the office contacts {number} students, "
    f"{TP} students who later leave are reached in time, "
    f"{FP} students are worried for nothing, "
    f"and {FN} students who later leave are missed."
)


st.header("Results per group")


def group_results(data):

    TP = (
        (data["left"] == 1)
        & (data["contacted"] == True)
    ).sum()

    FP = (
        (data["left"] == 0)
        & (data["contacted"] == True)
    ).sum()

    FN = (
        (data["left"] == 1)
        & (data["contacted"] == False)
    ).sum()

    TN = (
        (data["left"] == 0)
        & (data["contacted"] == False)
    ).sum()

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    return TP, FP, FN, TN, precision, recall


domestic = val[val["international"] == 0]
international = val[val["international"] == 1]

domestic_results = group_results(domestic)
international_results = group_results(international)


group_table = pd.DataFrame(
    {
        "Group": [
            "Domestic",
            "International"
        ],
        "Reached in time": [
            domestic_results[0],
            international_results[0]
        ],
        "Worried for nothing": [
            domestic_results[1],
            international_results[1]
        ],
        "Missed": [
            domestic_results[2],
            international_results[2]
        ],
        "Correctly left alone": [
            domestic_results[3],
            international_results[3]
        ],
        "Precision": [
            domestic_results[4],
            international_results[4]
        ],
        "Recall": [
            domestic_results[5],
            international_results[5]
        ]
    }
)

st.dataframe(
    group_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Precision": st.column_config.NumberColumn(
            format="%.1%%"
        ),
        "Recall": st.column_config.NumberColumn(
            format="%.1%%"
        )
    }
)


st.subheader("Domestic students")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Reached in time",
    domestic_results[0]
)

c2.metric(
    "Worried for nothing",
    domestic_results[1]
)

c3.metric(
    "Missed",
    domestic_results[2]
)


st.subheader("International students")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Reached in time",
    international_results[0]
)

c2.metric(
    "Worried for nothing",
    international_results[1]
)

c3.metric(
    "Missed",
    international_results[2]
)
