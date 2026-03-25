"""
Módulo de construcción del modelo y definición de funciones de pérdida para
la segmentación multiclase.

Se implementa:
- Construcción del modelo de segmentación basado en U-Net++.
- Inicialización configurable del encoder y sus pesos preentrenados.
- Definición de una función de pérdida compuesta para la segmentación.

--------------------------------------------------------------------------------

Variables esperadas (config.py):
- NUM_CLASSES   : número total de clases.
- DEVICE        : 'cpu' o 'cuda'.
"""

import torch.nn as nn
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

from config import *

def build_model(encoder="resnet34", pesos=None):
    """
    Construye un modelo de segmentación basado en U-Net++.

    --------------------------------------------------------------------------------

    Parámetros:
    - encoder: Nombre del encoder backbone utilizado por U-Net++ (str, opcional).
    - pesos: Pesos preentrenados del encoder (str o None, opcional).

    Retorna:
    - model: Modelo U-Net++ configurado para segmentación (torch.nn.Module).
    """
    model = smp.UnetPlusPlus(
        encoder_name = encoder,
        encoder_weights = pesos,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model

def build_loss():
    """
    Define y retorna una función de pérdida compuesta para la segmentación multiclase.

    --------------------------------------------------------------------------------

    Retorna:
    - loss_fn: Función de pérdida que recibe (pred, target) y retorna un escalar (callable).
    """
    class_weights = torch.tensor(
        [1.0, 1.5, 1.5, 2.0, 4.0, 6.0],      # Ponderadores
        dtype=torch.float,
    ).to(DEVICE)

    ce = nn.CrossEntropyLoss(weight=class_weights)
    dice = DiceLoss(mode="multiclass", from_logits=True, smooth=1.0)
    # focal = FocalLoss(mode="multiclass", gamma=1.5)

    def loss_fn(pred, target):
        return 0.5 * dice(pred, target) + 0.5 * ce(pred, target)   # Función de Pérdida Compuesta

    return loss_fn