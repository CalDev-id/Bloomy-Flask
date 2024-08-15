from typing import Union
from fastapi import FastAPI, UploadFile, File, HTTPException
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

#model fish or shrimp
#=======================================================================================================================
try:
    model = load_model("models/my_model.keras")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None



@app.post("/predict-fish-or-shrimp/")
async def predictFishorShrimp(file: UploadFile = File(...)):
    # Dimensi input yang diharapkan oleh model
    IMG_WIDTH, IMG_HEIGHT = 150, 150
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
    
#model fish Grading
#=======================================================================================================================
try:
    model_fishgrading = load_model("models/marine_grading_fish.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model_fishgrading = None

@app.post("/predict-fishgrading/")
async def predictFishGrading(file: UploadFile = File(...)):
    IMG_WIDTH, IMG_HEIGHT = 160, 160

    if model_fishgrading is None:
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
        classes = model_fishgrading.predict(img_array, batch_size=1)
        
        class_list = ['A', 'B', "C"]
        predicted_class = class_list[np.argmax(classes[0])]

        # Mengembalikan hasil prediksi sebagai JSON response
        # return JSONResponse(content={"predicted_class": int(predicted_class)})
        return JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

#model Shrimp Grading
#=======================================================================================================================
try:
    model_fishgrading = load_model("models/marine_grading_shrimp.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model_fishgrading = None

@app.post("/predict-shrimpgrading/")
async def predictFishGrading(file: UploadFile = File(...)):
    IMG_WIDTH, IMG_HEIGHT = 160, 160

    if model_fishgrading is None:
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
        classes = model_fishgrading.predict(img_array, batch_size=1)
        
        class_list = ['A', 'B', "C"]
        predicted_class = class_list[np.argmax(classes[0])]

        # Mengembalikan hasil prediksi sebagai JSON response
        # return JSONResponse(content={"predicted_class": int(predicted_class)})
        return JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")    
    
#model fish or shrimp grading
#=======================================================================================================================
# Model Fish or Shrimp
try:
    model = load_model("models/my_model.keras")
    print("Fish or Shrimp model loaded successfully.")
except Exception as e:
    print(f"Error loading Fish or Shrimp model: {e}")
    model = None

# Model Fish Grading
try:
    model_fishgrading = load_model("models/marine_grading_fish.h5")
    print("Fish grading model loaded successfully.")
except Exception as e:
    print(f"Error loading Fish grading model: {e}")
    model_fishgrading = None

# Model Shrimp Grading
try:
    model_shrimpgrading = load_model("models/marine_grading_shrimp.h5")
    print("Shrimp grading model loaded successfully.")
except Exception as e:
    print(f"Error loading Shrimp grading model: {e}")
    model_shrimpgrading = None


@app.post("/marine-grading/")
async def marineGrading(file: UploadFile = File(...)):
    IMG_WIDTH, IMG_HEIGHT = 150, 150
    if model is None:
        raise HTTPException(status_code=500, detail="Fish or Shrimp model is not loaded")

    try:
        # Membaca file gambar
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Mengubah gambar menjadi format RGB jika tidak dalam format tersebut
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Memproses gambar
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Melakukan prediksi
        classes = model.predict(img_array)
        predicted_class = 'Ikan' if np.argmax(classes[0]) == 0 else 'Udang'

        if predicted_class == 'Ikan':
            if model_fishgrading is None:
                raise HTTPException(status_code=500, detail="Fish grading model is not loaded")
            grading_model = model_fishgrading
            IMG_WIDTH, IMG_HEIGHT = 160, 160
            class_list = ['A', 'B', 'C']
        else:
            if model_shrimpgrading is None:
                raise HTTPException(status_code=500, detail="Shrimp grading model is not loaded")
            grading_model = model_shrimpgrading
            IMG_WIDTH, IMG_HEIGHT = 160, 160
            class_list = ['A', 'B', 'C']

        # Resize image for grading model
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Melakukan prediksi grading
        grading_classes = grading_model.predict(img_array)
        grading_result = class_list[np.argmax(grading_classes[0])]

        # Mengembalikan hasil prediksi dan grading sebagai JSON response
        return JSONResponse(content={"predicted_class": predicted_class, "grading_result": grading_result})

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

#model sail decision
#=======================================================================================================================

try:
    model_sailDecision = load_model("models/marine_sail_decision.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model_sailDecision = None

@app.post("/sail-decision/")
async def sailDecision(input_data: list[list[int]]):
    if model_sailDecision is None:
        raise HTTPException(status_code=500, detail="Sail Decision model is not loaded")

    try:
        # Memastikan input_data adalah array numpy dengan bentuk (1, 4)
        input_array = np.array(input_data)

        # Melakukan prediksi
        prediction = model_sailDecision.predict(input_array)
        result = prediction[0][0]  # Mengambil hasil prediksi pertama

        # Menentukan apakah aman untuk berlayar atau tidak
        decision = 'Aman Untuk Berlayar' if result >= 0.5 else 'Tidak Aman Untuk Berlayar'
        binary_label = 1 if result >= 0.5 else 0

        # Mengembalikan hasil prediksi dan keputusan sebagai JSON response
        return JSONResponse(content={
            "predicted_class": int(np.argmax(prediction, axis=1)[0]),
            "decision": decision,
            "binary_label": binary_label
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)