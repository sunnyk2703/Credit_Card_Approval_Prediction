document.getElementById("predictionForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const data = {
        Gender: document.getElementById("Gender").value,
        Married: document.getElementById("Married").value,
        Dependents: document.getElementById("Dependents").value || "0",
        Education: document.getElementById("Education").value,
        Self_Employed: document.getElementById("Self_Employed").value,
        ApplicantIncome: document.getElementById("ApplicantIncome").value,
        CoapplicantIncome: document.getElementById("CoapplicantIncome").value,
        LoanAmount: document.getElementById("LoanAmount").value,
        Loan_Amount_Term: document.getElementById("Loan_Amount_Term").value,
        Credit_History: document.getElementById("Credit_History").value,
        Property_Area: document.getElementById("Property_Area").value
    };

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        const resultBox = document.getElementById("result");

        if (result.prediction === "Approved") {
            resultBox.innerHTML = "✅ Credit Card Approved";
            resultBox.className = "approved";
        } else {
            resultBox.innerHTML = "❌ Credit Card Rejected";
            resultBox.className = "rejected";
        }

    } catch (error) {
        document.getElementById("result").innerHTML =
            "Server Error!";
    }

});