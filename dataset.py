"""
Módulo de utilidades de preprocesamiento y dataset para la segmentación multiclase.

Se implementa:
- Funciones de letterbox y unletterbox para redimensionar imágenes y máscaras
  manteniendo la relación de aspecto.
- Definición de un Dataset compatible con PyTorch para la segmentación multiclase.

--------------------------------------------------------------------------------

Variables esperadas (config.py):
- IMAGE_SIZE         : tamaño de entrada del modelo.
- COLOR_MAP          : diccionario RGB a índice de clase.
- EXTS               : extensiones válidas de imágenes/máscaras.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from config import *

def letterbox(image, target_size, is_mask=False):
    """
    Redimensiona una imagen manteniendo su relación de aspecto y aplica
    padding para obtener una imagen cuadrada de tamaño (target_size x target_size).

    --------------------------------------------------------------------------------

    Parámetros:
    - image: Imagen de entrada en formato HxWxC (np.ndarray).
    - target_size: Tamaño deseado de la imagen final cuadrada (int).
    - is_mask: Indica si la imagen corresponde a una máscara (bool, opcional).

    Retorna:
    - image_padded: Imagen redimensionada y con padding (np.ndarray).
    - meta: Datos necesarios para revertir el letterbox (dict).
        - scale: Factor de escalado.
        - pad: Padding aplicado (top, bottom, left, right).
        - orig_shape: Tamaño original de la imagen.
    """
    h, w = image.shape[:2]

    # Escala basada en el lado mayor para mantener proporción
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    # Interpolación adecuada según si es máscara o no (se evitan clases nuevas en máscaras)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    image_resized = cv2.resize(image, (nw, nh), interpolation=interp)

    # Cálculo del padding para centrar la imagen
    pad_top = (target_size - nh) // 2
    pad_bottom = target_size - nh - pad_top
    pad_left = (target_size - nw) // 2
    pad_right = target_size - nw - pad_left

    image_padded = cv2.copyMakeBorder(
        image_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=0
    )

    meta = {
        "scale": scale,
        "pad": (pad_top, pad_bottom, pad_left, pad_right),
        "orig_shape": (h, w)
    }

    return image_padded, meta

def unletterbox_mask(mask, meta):
    """
    Revierte el proceso de letterbox sobre una máscara,
    eliminando el padding y restaurando su tamaño original.

    --------------------------------------------------------------------------------

    Parámetros:
    - mask: Máscara predicha en tamaño letterbox (np.ndarray).
    - meta: Datos generados por la función letterbox (dict).

    Retorna:
    - mask: Máscara reescalada al tamaño original (np.ndarray).
    """
    pad_top, pad_bottom, pad_left, pad_right = meta["pad"]
    h, w = meta["orig_shape"]

    # Eliminación del padding
    mask = mask[
        pad_top: mask.shape[0] - pad_bottom,
        pad_left: mask.shape[1] - pad_right
    ]

    # Reescalado al tamaño original
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        """
        Parámetros:
        - img_dir: Directorio que contiene las imágenes (str o Path).
        - mask_dir: Directorio que contiene las máscaras (str o Path).
        - transform: Augmentations (callable, opcional).
        """
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # Lista de imágenes válidas según extensiones permitidas
        self.images = [
            p.name for p in self.img_dir.iterdir()
            if p.suffix.lower() in EXTS
        ]

    def __len__(self):
        """Retorna el número total de muestras del dataset."""
        return len(self.images)

    def mask_to_class(self, mask):
        """
        Convierte una máscara RGB a una máscara de clases enteras,
        usando un mapa de colores definido en COLOR_MAP (config).

        --------------------------------------------------------------------------------
        Parámetros:
        - mask: Máscara RGB (HxWx3) (np.ndarray).

        Retorna:
        - class_mask: Máscara 2D con índices de clase (np.ndarray).
        """
        h, w, _ = mask.shape
        class_mask = np.zeros((h, w), dtype=np.int64)

        for rgb, cls in COLOR_MAP.items():
            matches = np.all(mask == np.array(rgb), axis=-1)
            class_mask[matches] = cls

        return class_mask

    def __getitem__(self, idx):
        """
        Retorna una muestra del dataset.

        --------------------------------------------------------------------------------
        Retorna:
        - image: Imagen normalizada (C x H x W) (torch.Tensor).
        - mask: Máscara de clases (H x W) (torch.Tensor).
        """
        img_name = self.images[idx]
        base = Path(img_name).stem

        img_path = self.img_dir / img_name

        # Búsqueda de la máscara con extensiones permitidas
        mask_path = None
        for ext in EXTS:
            p = self.mask_dir / f"{base}{ext}"
            if p.exists():
                mask_path = p
                break

        if mask_path is None:
            raise FileNotFoundError(f"Máscara no encontrada para {img_name}")

        # Lectura y conversión a RGB
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Letterbox
        image, meta = letterbox(image, IMAGE_SIZE, is_mask=False)
        mask, _ = letterbox(mask, IMAGE_SIZE, is_mask=True)

        # Conversión RGB a clases
        mask = self.mask_to_class(mask)

        # Data augmentation
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        # Conversión a tensores
        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask