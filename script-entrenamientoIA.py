# =====================================================
# 1️⃣ INSTALAR DEPENDENCIAS
# =====================================================

!pip install -q tensorflow rasterio

# =====================================================
# 2️⃣ IMPORTS
# =====================================================

import os
import numpy as np
import tensorflow as tf
import rasterio
from sklearn.model_selection import train_test_split

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# =====================================================
# 3️⃣ RUTA DATASET (YA DESCARGADO)
# =====================================================

DATA_DIR = "EuroSAT_MS"  # carpeta con las clases
IMG_SIZE = 224
BATCH_SIZE = 32

# =====================================================
# 4️⃣ LEER ARCHIVOS .tif
# =====================================================

image_paths = []
labels = []
class_names = sorted(os.listdir(DATA_DIR))

for idx, class_name in enumerate(class_names):
    class_folder = os.path.join(DATA_DIR, class_name)
    if os.path.isdir(class_folder):
        for file in os.listdir(class_folder):
            if file.endswith(".tif"):
                image_paths.append(os.path.join(class_folder, file))
                labels.append(idx)

print("Clases:", class_names)
print("Total imágenes:", len(image_paths))

# =====================================================
# 5️⃣ CONVERTIR A FOREST VS NON-FOREST
# =====================================================

FOREST_INDEX = class_names.index("Forest")
labels = np.array(labels)
labels = (labels == FOREST_INDEX).astype(np.float32)

# =====================================================
# 6️⃣ TRAIN / VALID SPLIT
# =====================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

# =====================================================
# 7️⃣ FUNCIÓN PARA CARGAR .tif
# =====================================================

def load_tif(path, label):
    path = path.numpy().decode("utf-8")
    
    with rasterio.open(path) as src:
        img = src.read()  # shape: (bands, H, W)
    
    img = np.transpose(img, (1, 2, 0))  # (H, W, bands)
    
    # Si tiene más de 3 bandas, tomar solo primeras 3
    if img.shape[2] > 3:
        img = img[:, :, :3]
    
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    
    return img, label

def tf_wrapper(path, label):
    img, label = tf.py_function(load_tif, [path, label], [tf.float32, tf.float32])
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    label.set_shape(())
    return img, label

# =====================================================
# 8️⃣ CREAR DATASETS
# =====================================================

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =====================================================
# 9️⃣ MODELO EfficientNetB0
# =====================================================

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# =====================================================
# 🔥 FASE 1
# =====================================================

model.fit(train_ds, validation_data=val_ds, epochs=5)

# =====================================================
# 🔥 FINE TUNING
# =====================================================

base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.fit(train_ds, validation_data=val_ds, epochs=5)

# =====================================================
# 📊 EVALUAR
# =====================================================

results = model.evaluate(val_ds)

print("\nRESULTADOS:")
print("Loss:", results[0])
print("Accuracy:", results[1])
print("AUC:", results[2])

# =====================================================
# 💾 GUARDAR MODELO
# =====================================================

model.save("eurosat_forest_model_ms.keras")
print("\nModelo guardado correctamente.")