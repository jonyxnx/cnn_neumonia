import os
from flask import Flask, request, redirect, url_for, render_template
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet


model = torch.load('models/model.pth')
model = EfficientNet.from_pretrained('efficientnet-b0')
num_features = model._fc.in_features
model._fc = torch.nn.Linear(num_features, 2)  
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.5, .5, .5], std=[0.5, .5, .5])  
])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    image_path = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        image = request.files['file']
        
        if image.filename == '':
            return redirect(request.url)

        img = Image.open(image).convert('RGB') 
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        img.save(image_path)

        if prediction == 1:
            prediction_message = 'Normal lungs'
        else:
            prediction_message = 'Pneumonia'

        return render_template('upload.html', prediction=prediction_message, image_path=image_path)

    return render_template('upload.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)

