from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io
import torch.nn as nn
from flask import render_template
import base64
import os

app = Flask(__name__)

# 🔷 Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 2)
)

model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()

labels = ['Benign', 'Malignant']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    uploaded_image_data = None
    confidence = None

    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        img_bytes = file.read()

        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, predicted = torch.max(probs, 1)

        prediction = labels[predicted.item()]
        confidence = conf.item() * 100  # in percent

        buffered = io.BytesIO()
        image.save(buffered, format='JPEG')
        uploaded_image_data = base64.b64encode(buffered.getvalue()).decode()
    return render_template(
        'index.html',
        prediction=prediction,
        confidence=round(confidence, 2) if confidence else None,
        uploaded_image_data=uploaded_image_data
    )



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, predicted = torch.max(probs, 1)

        result = labels[predicted.item()]
        confidence = conf.item() * 100
        return jsonify({"prediction": result, "confidence": round(confidence, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
