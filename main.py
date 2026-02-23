from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os

# ============================================
# CONFIGURACIÓN
# ============================================

app = FastAPI(title="API IA - Clasificación Forest / Non-Forest")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# CARGA DEL MODELO
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 USA EL MODELO NUEVO
MODEL_PATH = os.path.join(BASE_DIR, "eurosat_forest_model_ms.keras")

forest_model = None

def load_model():
    global forest_model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"❌ Modelo no encontrado en {MODEL_PATH}")

    try:
        print("🔄 Cargando modelo EfficientNet...")
        forest_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        raise RuntimeError(f"Error cargando modelo: {str(e)}")

load_model()

# ============================================
# PREPROCESAMIENTO (IGUAL QUE ENTRENAMIENTO)
# ============================================

IMG_SIZE = 224

def preprocess_image(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    arr = np.array(image).astype(np.float32)

    # 🔥 IMPORTANTÍSIMO
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    arr = np.expand_dims(arr, axis=0)

    return arr

# ============================================
# ENDPOINT PRINCIPAL
# ============================================

@app.post("/ia/predict-forest")
async def predict_forest(file: UploadFile = File(...)):

    if forest_model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        contents = await file.read()
        x = preprocess_image(contents)

        prediction = float(forest_model.predict(x, verbose=0)[0][0])

        THRESHOLD = 0.5

        is_forest = prediction > THRESHOLD

        return {
            "label": "forest" if is_forest else "no_forest",
            "probabilidad_forest": round(prediction, 6),
            "threshold": THRESHOLD,
            "modelo": "EuroSAT EfficientNetB0",
            "nota": "Clasifica presencia de bosque, no cambio temporal"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# ============================================
# HEALTH CHECK
# ============================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo_cargado": forest_model is not None
    }