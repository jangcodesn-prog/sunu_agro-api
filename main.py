from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os

# ==============================
# INITIALISATION
# ==============================

app = FastAPI(title="Sunu Agro AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# CONFIG
# ==============================

MODEL_PATH = "modele_feuille.h5"
model = None  # lazy loading

# ==============================
# CHARGEMENT LAZY DU MODÈLE
# ==============================

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

# ==============================
# CLASSES
# ==============================

class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Healthy"
]

# ==============================
# INFOS MALADIES
# ==============================

disease_info = {
    "Tomato___Early_blight": {
        "description": "Maladie fongique causée par Alternaria solani.",
        "recommendation": "Utiliser un fongicide à base de cuivre et retirer les feuilles infectées."
    },
    "Tomato___Late_blight": {
        "description": "Maladie causée par Phytophthora infestans.",
        "recommendation": "Appliquer un traitement antifongique et éviter l'humidité excessive."
    },
    "Tomato___Healthy": {
        "description": "La plante est en bonne santé.",
        "recommendation": "Aucune action nécessaire."
    }
}

# ==============================
# ROUTE TEST
# ==============================

@app.get("/")
def home():
    return {"message": "API IA Sunu Agro opérationnelle 🌱"}

# ==============================
# ROUTE PRÉDICTION
# ==============================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(status_code=400, detail="Aucun fichier envoyé")

    try:
        # lazy load modèle
        model_local = get_model()

        # lecture image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # preprocessing
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # prédiction
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