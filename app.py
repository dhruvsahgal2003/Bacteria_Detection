from flask import Flask, request, jsonify, send_from_directory, url_for
from inference_sdk import InferenceHTTPClient
import os
import uuid
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="aBYOj9RAujbYmWZ7UDTX"
)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

   
    temp_path = os.path.join('uploads', file.filename)
    file.save(temp_path)

  
    result = CLIENT.infer(temp_path, model_id="bacteria-detection-jihrt/2")

    
    if isinstance(result, dict) and 'predictions' in result:
        predictions = result['predictions']
    else:
        return jsonify({'error': 'Unexpected response format from model'}), 500

   
    image = Image.open(temp_path)
    draw = ImageDraw.Draw(image)
    
    # Load a font
    font = ImageFont.load_default()

    for pred in predictions:
        x = pred['x']
        y = pred['y']
        width = pred['width']
        height = pred['height']
        confidence = pred['confidence']
        class_name = pred['class']
        
        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2
        
        draw.rectangle([left, top, right, bottom], outline='red', width=2)
        
      
        label = f"{class_name}: {confidence:.2f}"
        
       
        text_size = draw.textsize(label, font=font)
        text_x = left
        text_y = top - text_size[1]
        
       
        draw.rectangle([text_x, text_y, text_x + text_size[0], text_y + text_size[1]], fill='red')
        
       
        draw.text((text_x, text_y), label, fill='white', font=font)

  
    output_filename = f"{uuid.uuid4().hex}.jpg"
    output_path = os.path.join('outputs', output_filename)
    image.save(output_path)

    # Remove the temporary input file
    os.remove(temp_path)

    # Return the URL to the processed image
    return jsonify({'image_url': url_for('get_image', filename=output_filename), 'result': result})

@app.route('/outputs/<filename>')
def get_image(filename):
    return send_from_directory('outputs', filename)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    app.run(debug=True)
