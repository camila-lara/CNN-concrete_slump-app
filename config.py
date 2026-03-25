"""
Archivo de configuración general del proyecto.

Se definen:
- Rutas del dataset, archivo Excel de datos y carpetas de salida.
- Parámetros de entrenamiento del modelo de segmentación.
- Configuración del encoder y del dispositivo de cómputo.
- Definición de clases de segmentación y su correspondencia en máscaras RGB.
- Constantes físicas utilizadas en cálculos posteriores.

Estructura esperada del dataset!!!!
DATASET_DIR/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
"""

from pathlib import Path
import torch

# --- RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
# DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR = BASE_DIR / "dataset_10cm"

TRAIN_IMG_DIR = DATASET_DIR / "images" / "train"
TRAIN_MASK_DIR = DATASET_DIR / "masks" / "train"

VAL_IMG_DIR = DATASET_DIR / "images" / "val"
VAL_MASK_DIR = DATASET_DIR / "masks" / "val"

TEST_IMG_DIR = DATASET_DIR / "images" / "test"
TEST_MASK_DIR = DATASET_DIR / "masks" / "test"

OUTPUT_DIR = BASE_DIR / "outputs"
# MODEL_PATH = OUTPUT_DIR / "modelo_conos.pth"
MODEL_PATH = OUTPUT_DIR / "modelo_conos_10cm.pth"

EXCEL_PATH = BASE_DIR / "data" / "Detalle200.xlsx"

# --- CARACTERISTICAS ENTRENAMIENTO ---
IMAGE_SIZE = 1024
BATCH_SIZE = 2
EPOCHS = 100

ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- LR DINAMICO (O NO) ---
LR = 1e-4
# STEP_SIZE = 
# GAMMA = 

# --- CLASES ---
CLASE_FONDO = 0
CLASE_ASENTAMIENTO = 1
CLASE_CONO = 2
CLASE_HUINCHA = 3
CLASE_BARRA = 4
CLASE_MARCA_10 = 5

NUM_CLASSES = 6

COLOR_MAP = {
    (0, 0, 0): 0,           # Fondo
    (255, 0, 0): 1,         # Asentamiento
    (0, 255, 0): 2,         # Cono
    (0, 0, 255): 3,         # Huincha
    (255, 255, 0): 4,       # Barra
    (255, 0, 255): 5,       # Marca 10
}

CLASS_NAMES = {
    CLASE_FONDO: "Fondo",
    CLASE_ASENTAMIENTO: "Asentamiento",
    CLASE_CONO: "Cono",
    CLASE_HUINCHA: "Huincha",
    CLASE_BARRA: "Barra",
    CLASE_MARCA_10: "Marca 10"
}

# --- EXTRAS ---
HUINCHA_REAL_ANCHO_CM = 2.5
USE_WARP_HUINCHA = True   # Aplicar distorsion de imagen basada en huincha
WARP_OUT_W = 400
WARP_OUT_H = 1400


EXTS = [".png", ".jpg", ".jpeg"]