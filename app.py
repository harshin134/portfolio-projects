from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ✅ ADD THIS (store logs)
logs = []

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        user_input = request.form["text"]

        # Convert input to vector
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "⚠️ Threat Detected"
        else:
            result = "✅ Safe"

        # ✅ ADD THIS (save logs)
        logs.append({
            "text": user_input,
            "result": result
        })

    # ✅ PASS logs to HTML
    return render_template("index.html", result=result, logs=logs)

if __name__ == "__main__":
    app.run(debug=True)
