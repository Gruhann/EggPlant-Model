import numpy as np
import tensorflow as tf
import cv2
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Dict

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

# Load model at startup
interpreter = None

# Class names for the model
quality_class_names = ["good", "semigood", "bad"]
# Set minimum confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.1

class PredictionResponse(BaseModel):
    quality: str
    accuracy: float
    has_eggplant: bool
    message: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global interpreter
    try:
        # Load model
        interpreter = tf.lite.Interpreter(model_path="Advanced.tflite")
        interpreter.allocate_tensors()
        
        # Print model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Model loaded successfully")
        print("Input details:", input_details)
        print("Output details:", output_details)
    except Exception as e:
        print(f"Error loading model: {str(e)}")

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True):
    """Resize image and pad for YOLOv8 input"""
    shape = img.shape[:2]  # current shape [height, width]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
        
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
        
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        
    if scale_fill:  # stretch
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios
        
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, ratio, (dw, dh)

def process_output(output_data, input_shape, original_shape, confidence_threshold=0.3):
    """Process YOLOv8 TFLite output and get the highest confidence detection and its class"""
    print("Processing output shape:", output_data.shape)
    output = output_data[0]  # First output is typically the detection data
    
    # YOLOv8 output shape is [batch, boxes, 85] 
    # where 85 = 4 (box coords) + 1 (objectness) + 80 (class scores)
    # or in some exported models it's [batch, boxes, 4+1+num_classes]
    
    num_boxes = output.shape[0]
    print(f"Number of detections: {num_boxes}")
    box_data = []
    max_confidence = 0
    predicted_class = None
    has_eggplant = False
    
    for i in range(num_boxes):
        confidence = output[i][4]  # Objectness score
        if confidence > confidence_threshold:
            has_eggplant = True
            class_id = np.argmax(output[i][5:])
            class_score = output[i][5 + class_id]
            total_confidence = float(confidence * class_score)
            
            if total_confidence > max_confidence:
                max_confidence = total_confidence
                predicted_class = quality_class_names[class_id] if class_id < len(quality_class_names) else f"unknown"
    
    return has_eggplant, predicted_class, max_confidence

def predict_quality(img):
    global interpreter
    
    if interpreter is None:
        return False, "error", 0.0, "Model not loaded"
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # YOLOv8 expects RGB images, convert if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    # Get input shape
    input_shape = input_details[0]['shape'][1:3]  # [height, width]
    
    # Preprocess image using YOLOv8 letterbox function
    preprocessed_img, ratio, padding = letterbox(img_rgb, input_shape)
    
    # Normalize and prepare input
    input_data = preprocessed_img.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    print("Input shape:", input_data.shape)
    print("Input value range:", input_data.min(), "to", input_data.max())
    
    # Set the input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Model output shape:", output_data.shape)
    print("First few values:", output_data[0, :5])  # Print first 5 values
    
    # Process results
    original_shape = img.shape[:2]
    has_eggplant, predicted_class, confidence = process_output(
        output_data, input_shape, original_shape, CONFIDENCE_THRESHOLD
    )
    
    if not has_eggplant:
        return False, None, 0.0, "No eggplant detected in the image"
    
    if predicted_class is None:
        return True, None, 0.0, "Eggplant detected but quality classification failed"
    
    return True, predicted_class, confidence, "Success"

@app.get("/")
def read_root():
    return {"message": "Eggplant Quality Detection API is running. Use /predict endpoint to analyze images."}

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        
        # Convert to numpy array for OpenCV
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image format. Please upload a valid image."}
            )

        # Predict quality
        has_eggplant, quality, accuracy, message = predict_quality(img)
        
        if not has_eggplant:
            return {
                "quality": "unknown",
                "accuracy": 0.0,
                "has_eggplant": False,
                "message": message
            }
        
        if quality is None:
            return {
                "quality": "unknown",
                "accuracy": 0.0,
                "has_eggplant": True,
                "message": message
            }
        
        return {
            "quality": quality,
            "accuracy": accuracy,
            "has_eggplant": True,
            "message": "Success"
        }
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# For local development
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 