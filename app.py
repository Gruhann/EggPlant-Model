import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import cv2

app = FastAPI(title="Eggplant Quality Detection API",
              description="API for detecting quality of eggplants using Keras model",
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
model = None

# Class names for the model
quality_class_names = ["good", "semigood", "bad"]

class PredictionResponse(BaseModel):
    quality: str
    accuracy: float
    has_eggplant: bool
    message: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        # Load model
        model = load_model('82accur.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

def predict_quality(img):
    global model
    
    if model is None:
        return False, "error", 0.0, "Model not loaded"
    
    try:
        # Resize image to target size
        img = cv2.resize(img, (256, 256))
        
        # Convert to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Convert to array and normalize
        img_array = image.img_to_array(img_rgb)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class_idx])
        
        # Get class name
        predicted_class = quality_class_names[predicted_class_idx]
        
        return True, predicted_class, confidence, "Success"
        
    except Exception as e:
        print(f"Error in predict_quality: {str(e)}")
        return False, "error", 0.0, f"Error during prediction: {str(e)}"

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

# Previous code (commented out)
# import numpy as np
# import tensorflow as tf
# import cv2
# import os
# import uvicorn
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional, Union, Dict
# ... (rest of the previous code remains commented) 