import numpy as np
import tensorflow as tf
import cv2
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from typing import Optional

app = FastAPI(title="Eggplant Quality Detection API",
              description="API for detecting quality of eggplants using YOLOv8 model",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path: str) -> tuple[Optional[dict], Optional[str]]:
    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path="Advanced.tflite")
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare the input image
        img = cv2.imread(image_path)
        if img is None:
            return None, "Could not read image"

        # Get the input shape
        input_shape = input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2]

        # Resize and preprocess the image
        img_resized = cv2.resize(img, (input_width, input_height))
        img_normalized = img_resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(img_normalized, axis=0)

        # Set the input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Process YOLOv8 output
        if len(output_data.shape) == 3 and output_data.shape[1] <= 85:
            outputs = np.transpose(output_data[0], (1, 0))
            img_height, img_width = img.shape[:2]
            detections = []
            conf_threshold = 0.25

            for i in range(len(outputs)):
                if outputs.shape[1] >= 6:
                    x, y, w, h = outputs[i][:4]
                    conf = outputs[i][4]
                    
                    if conf > conf_threshold:
                        class_id = int(np.argmax(outputs[i][5:]))
                        class_prob = outputs[i][5 + class_id]
                        
                        x1 = int((x - w/2) * img_width)
                        y1 = int((y - h/2) * img_height)
                        x2 = int((x + w/2) * img_width)
                        y2 = int((y + h/2) * img_height)
                        
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_width, x2)
                        y2 = min(img_height, y2)
                        
                        detections.append({
                            'class_id': class_id,
                            'confidence': float(conf),
                            'box': [x1, y1, x2, y2]
                        })

            # Draw bounding boxes on the image
            for det in detections:
                x1, y1, x2, y2 = det['box']
                color = (0, 255, 0) if det['class_id'] == 0 else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{'Healthy' if det['class_id'] == 0 else 'Diseased'}: {det['confidence']:.2f}"
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save the processed image
            output_path = os.path.join(UPLOAD_FOLDER, 'processed_' + os.path.basename(image_path))
            cv2.imwrite(output_path, img)

            # Convert image to base64
            with open(output_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Clean up
            os.remove(output_path)

            return {
                'has_eggplant': len(detections) > 0,
                'quality': 'good' if len(detections) > 0 and detections[0]['class_id'] == 0 else 'bad',
                'accuracy': float(detections[0]['confidence']) if len(detections) > 0 else 0,
                'message': 'Success' if len(detections) > 0 else 'No eggplant detected',
                'processed_image': encoded_image
            }, None

        return None, "Unexpected output format from model"

    except Exception as e:
        return None, str(e)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        # Save the uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(filepath, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the image
        result, error = process_image(filepath)
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        if error:
            raise HTTPException(status_code=500, detail=error)
            
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000) 