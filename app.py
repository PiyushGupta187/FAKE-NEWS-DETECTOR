from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# ✅ Load your saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# ✅ Home route — loads your HTML UI
@app.route('/')
def home():
    return render_template('index.html')


# ✅ Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_text = data.get('text', '')

    # Transform text using vectorizer
    transformed = vectorizer.transform([news_text])

    # Predict
    prediction = model.predict(transformed)[0]

    # Get confidence
    probabilities = model.predict_proba(transformed)[0]
    confidence = round(max(probabilities) * 100, 2)

    # 0=FAKE, 1=REAL
    result = 'FAKE' if prediction == 0 else 'REAL'

    return jsonify({'result': result, 'confidence': confidence})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)