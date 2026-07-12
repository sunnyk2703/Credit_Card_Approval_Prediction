from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model/model.pkl")
encoders = joblib.load("model/encoders.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)

        df = pd.DataFrame([data])

        categorical_columns = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area"
        ]

        for col in categorical_columns:
            df[col] = encoders[col].transform(df[col])

        df["ApplicantIncome"] = df["ApplicantIncome"].astype(float)
        df["CoapplicantIncome"] = df["CoapplicantIncome"].astype(float)
        df["LoanAmount"] = df["LoanAmount"].astype(float)
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype(float)
        df["Credit_History"] = df["Credit_History"].astype(float)

        prediction = model.predict(df)[0]

        prediction = encoders["Loan_Status"].inverse_transform([prediction])[0]
        print("Prediction:",prediction)
        if prediction == "Y":
            result = "Approved"
        else:
            result = "Rejected"

        return jsonify({"prediction": result})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"prediction": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)