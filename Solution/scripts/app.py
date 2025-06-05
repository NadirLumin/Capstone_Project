from flask import Flask, request, jsonify
import predictor

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': "Missing required 'text' field."}), 400
    input_text = data['text']
    try:
        output_text = predictor.predict(input_text)
        return jsonify({'input': input_text, 'output': output_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
