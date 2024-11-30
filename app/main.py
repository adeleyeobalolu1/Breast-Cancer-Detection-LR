import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def clean_data():
    # import data
    data = pd.read_csv("data\data.csv")

    # clean data
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


def add_side_bar():
    st.sidebar.header("Data Features")

    # Get the labels for the slider
    data = pd.read_csv("data\data.csv")
    data = data.drop(["Unnamed: 32", "id", "diagnosis"], axis=1)

    data_labels = data.columns.tolist()

    slider_labels = []
    for title in data_labels:
        words = title.split("_")
        if len(words) > 1:
            formatted_title = (
                " ".join(word.capitalize() for word in words[:-1]) + f" ({words[-1]})"
            )
        else:
            formatted_title = words[0].capitalize()
        slider_labels.append((formatted_title, title))

    # Create a data dictionary for the slider
    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
        )

    return input_dict


def scaled_values(input_data):
    data = clean_data()
    X = data.drop(["diagnosis"], axis=1)
    scaled_dict = {}
    for key, value in input_data.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled
    return scaled_dict


def radar_chart(data):
    input_data = scaled_values(data)
    labels = {
        " ".join(word.capitalize() for word in key.split("_")[:-1]) for key in data
    }
    categories = list(labels)
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[value for key, value in input_data.items() if "mean" in key],
            theta=categories,
            fill="toself",
            name="Mean Value",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[value for key, value in input_data.items() if "se" in key],
            theta=categories,
            fill="toself",
            name="Standard Error",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[value for key, value in input_data.items() if "worst" in key],
            theta=categories,
            fill="toself",
            name="Worst",
        )
    )

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    return fig


def prediction(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scalar.pkl", "rb"))

    input_array = (np.array(list(input_data.values()))).reshape(1, -1)

    scaled_array = scaler.transform(input_array)

    pred = model.predict(scaled_array)
    prob = model.predict_proba(scaled_array)
    prob_begnin = "{:.2f}%".format(prob[0][0] * 100)
    prob_malignant = "{:.2f}%".format(prob[0][1] * 100)

    st.header("Cell Cluster Prediction")
    st.write("The Cell Cluster is: ")

    if pred[0] == 0:
        st.subheader("Benign")
        st.write("The Probability of being Benign: ", prob_begnin)
    else:
        st.subheader("Malignant")
        st.write("The Probability of being Maligient: ", prob_malignant)

    st.write(
        "This app should not be used as substitute for professional diagnosis. But can assist in making medical diagnosis"
    )


def main():
    st.set_page_config(
        page_title="Breast Cancer Detection",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    input_data = add_side_bar()

    with st.container():
        st.title("Breast Cancer Detector")
        st.text(
            """This webapp detects breast cancer based on the breast tissue inputed data from a cytology lab. 
            The app uses a machine learning model to predict if a breast mass is BENIGN of MALIGNANT based on the cytology lab data
            Please connect the app to your lab to predict the samples"""
        )
    col1, col2 = st.columns([4, 2])

    with col1:
        radar = radar_chart(input_data)
        st.plotly_chart(radar)

    with col2:
        prediction(input_data)


if __name__ == "__main__":
    main()
