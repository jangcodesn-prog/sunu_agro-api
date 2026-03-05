from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI(title="Sunu Agro AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "modele_feuille.h5"
model = None

def get_model():
    global model
    if model is None:
        try:
            print("🔄 Chargement du modèle...")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Modèle chargé")
        except Exception as e:
            print("❌ Erreur chargement modèle :", e)
            raise e
    return model

class_names = [
    'Apple___Black_rot', 'Apple___healthy', 'Corn___Northern_Leaf_Blight',
    'Corn___Common_rust', 'Grape___Leaf_blight', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two_spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___healthy'
]

disease_info = {
    'Apple___Black_rot': {
        "description": "Maladie fongique causée par Botryosphaeria obtusa.",
        "recommendation": "Retirer les fruits infectés et appliquer un fongicide."
    },
    'Apple___healthy': {
        "description": "La plante est en bonne santé.",
        "recommendation": "Aucune action nécessaire."
    },
    'Corn___Northern_Leaf_Blight': {
        "description": "Maladie fongique causée par Exserohilum turcicum.",
        "recommendation": "Utiliser des semences résistantes et appliquer un fongicide."
    },
    'Corn___Common_rust': {
        "description": "Maladie fongique causée par Puccinia sorghi.",
        "recommendation": "Appliquer un fongicide adapté au maïs."
    },
    'Grape___Leaf_blight': {
        "description": "Maladie fongique affectant les feuilles de vigne.",
        "recommendation": "Appliquer un fongicide et améliorer la ventilation."
    },
    'Grape___healthy': {
        "description": "La plante est en bonne santé.",
        "recommendation": "Aucune action nécessaire."
    },
    'Potato___Early_blight': {
        "description": "Maladie fongique causée par Alternaria solani.",
        "recommendation": "Appliquer un fongicide et retirer les feuilles infectées."
    },
    'Potato___healthy': {
        "description": "La plante est en bonne santé.",
        "recommendation": "Aucune action nécessaire."
    },
    'Tomato___Bacterial_spot': {
        "description": "Maladie bactérienne causée par Xanthomonas.",
        "recommendation": "Utiliser un traitement à base de cuivre."
    },
    'Tomato___Early_blight': {
        "description": "Maladie fongique causée par Alternaria solani.",
        "recommendation": "Utiliser un fongicide à base de cuivre et retirer les feuilles infectées."
    },
    'Tomato___Late_blight': {
        "description": "Maladie causée par Phytophthora infestans.",
        "recommendation": "Appliquer un traitement antifongique et éviter l'humidité excessive."
    },
    'Tomato___Leaf_Mold': {
        "description": "Maladie fongique causée par Passalora fulva.",
        "recommendation": "Améliorer la ventilation et appliquer un fongicide."
    },
    'Tomato___Septoria_leaf_spot': {
        "description": "Maladie fongique causée par Septoria lycopersici.",
        "recommendation": "Retirer les feuilles infectées et appliquer un fongicide."
    },
    'Tomato___Spider_mites_Two_spotted_spider_mite': {
        "description": "Infestation d'acariens sur les feuilles de tomate.",
        "recommendation": "Utiliser un acaricide et augmenter l'humidité autour des plantes."
    },
    'Tomato___Target_Spot': {
        "description": "Maladie fongique causée par Corynespora cassiicola.",
        "recommendation": "Appliquer un fongicide et éviter l'excès d'humidité."
    },
    'Tomato___healthy': {
        "description": "La plante est en bonne santé.",
        "recommendation": "Aucune action nécessaire."
    }
}

@app.get("/")
def home():
    return {"message": "API IA Sunu Agro opérationnelle 🌱"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Aucun fichier envoyé")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Fichier doit être une image (jpg, png)")

    try:
        model_local = get_model()

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model_local.predict(image_array)
        confidence = float(np.max(predictions))
        label_index = int(np.argmax(predictions))
        label = class_names[label_index]

        description = disease_info[label]["description"]
        recommendation = disease_info[label]["recommendation"]

        return {
            "label": label,
            "confidence": confidence,
            "description": description,
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))