import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


# import data and clean
def clean_data():
    # import data
    data = pd.read_csv("data\data.csv")

    # clean data
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


# create model
def create_model(data):
    # assign features
    X = data.drop(["diagnosis"], axis=1)
    y = data["diagnosis"]

    # scale the features and split data
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scale, y, test_size=0.2, random_state=42
    )

    # train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # predict and check accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print("Model accuracy: ", {accuracy})
    print("Classification Report: ", {class_report})

    return model, scaler


def main():
    data = clean_data()

    model, scalar = create_model(data)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/scalar.pkl", "wb") as f:
        pickle.dump(scalar, f)


if __name__ == "__main__":
    main()
