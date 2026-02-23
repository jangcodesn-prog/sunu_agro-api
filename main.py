from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Autorise toutes les sources (ton téléphone)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# INITIALISATION DE L'APPLICATION
# ==============================

app = FastAPI(title="Sunu Agro AI API")

# ==============================
# CHARGEMENT DU MODÈLE
# ==============================

MODEL_PATH = "modele_feuille.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print("❌ Erreur chargement modèle :", e)
    raise e

# ==============================
# CLASSES (à adapter selon ton entraînement)
# ==============================

class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Healthy"
]

# ==============================
# INFORMATIONS SUR LES MALADIES
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
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
# ==============================
# ROUTE PRÉDICTION
# ==============================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(status_code=400, detail="Aucun fichier envoyé")

    try:
        # Lecture de l'image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Prétraitement (IMPORTANT : même taille que l'entraînement)
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Prédiction
        predictions = model.predict(image_array)
        confidence = float(np.max(predictions))
        label_index = int(np.argmax(predictions))
        label = class_names[label_index]

        # Récupération informations maladie
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
