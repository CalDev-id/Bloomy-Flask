from typing import Union
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

from groq import Groq
from typing import List, Dict, Any

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bloomy API"}

#load model
#=======================================================================================================================
try:
    model = load_model("models/my_model.keras")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Dimensi input yang diharapkan oleh model
IMG_WIDTH, IMG_HEIGHT = 150, 150

@app.post("/predict-fish-or-shrimp/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        # Membaca file gambar
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Mengubah gambar menjadi format RGB jika tidak dalam format tersebut
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Memproses gambar
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))

        #array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Menambahkan batch dimension
        img_array = img_array / 255.0

        # Melakukan prediksi
        # predictions = model.predict(img_array)
        # predicted_class = np.argmax(predictions, axis=1)[0]

        #Melakukan prediksi
        classes = model.predict(img_array, batch_size=1)
        
        class_list = ['Ikan', 'Udang']
        predicted_class = class_list[np.argmax(classes[0])]

        # Mengembalikan hasil prediksi sebagai JSON response
        # return JSONResponse(content={"predicted_class": int(predicted_class)})
        return JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)