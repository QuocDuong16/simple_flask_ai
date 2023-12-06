from flask import Flask, request, redirect, url_for
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from joblib import load
import os

app = Flask(__name__)

# Đường dẫn đến thư mục lưu trữ mô hình
model_path = "./model_distilbert/"
# Tải label encoder
label_encoder_category = load(model_path + 'label_encoder_category.joblib')
# Tải mô hình DistilBERT
model = DistilBertForSequenceClassification.from_pretrained(model_path)
# Tải tokenizer của DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Hàm dự đoán category từ name và description
def predict_category(name, description):
    # Kết hợp name và description
    text = name + ' ' + description
    # Tokenize và chuyển đổi dữ liệu thành định dạng đầu vào của DistilBERT
    inputs = tokenizer(text, max_length=50, truncation=True, padding=True, return_tensors='pt', dtype=torch.long)
    # Dự đoán category
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
    # Chuyển ngược lại từ số sang nhãn
    predicted_category = label_encoder_category.inverse_transform([predicted_label])[0]
    return predicted_category

@app.route('/api', methods=['GET'])
def predict():
    name = request.args.get('name', '')
    description = request.args.get('description', '')
    
    if name and description:
        predicted_category = predict_category(name, description)
        # Chuyển hướng đến URL mới với tham số 'category'
        return redirect(url_for('result', category=predicted_category))

    return 'Invalid input. Please provide both "name" and "description" parameters.', 400

@app.route('/api/result', methods=['GET'])
def result():
    predicted_category = request.args.get('category', '')
    return f'Result: {predicted_category}'

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
