"""
Cálculo automático de asentamiento de cono a partir de segmentación multiclase y geometría.

Flujo general:
1. Carga del modelo de segmentación entrenado.
2. Inferencia sobre imágenes de los conjuntos train / val / test.
3. Detección de blobs relevantes (huincha, barra y marca de 10 cm).
4. Cálculo de escala píxel-centímetro usando la marca de los 10 cm.
5. Cálculo del asentamiento (en píxeles y centímetros).
6. Extracción de múltiples features geométricas y relacionales.
7. Comparación con mediciones reales provenientes de un archivo Excel.
8. Cálculo de métricas base (MAE, RMSE, sesgo).
9. Entrenamiento de una regresión correctora usando solo features del conjunto train.
10. Aplicación de la corrección a todos los splits y evaluación final.
11. Exportación de resultados a CSV.

--------------------------------------------------------------------------------
Variables esperadas (config.py):
- DEVICE                                            : 'cpu' o 'cuda'.
- IMAGE_SIZE                                        : tamaño de entrada del modelo.
- MODEL_PATH                                        : ruta del modelo entrenado.
- ENCODER_NAME                                      : encoder del modelo.
- ENCODER_WEIGHTS                                   : pesos del encoder.
- CLASE_BARRA, CLASE_HUINCHA, CLASE_MARCA_10        : índices de clases relevantes.
- HUINCHA_REAL_ANCHO_CM                             : Ancho de la huincha en cm.
- TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR          : Rutas de imágenes del dataset.
- EXTS                                              : extensiones válidas de imágenes/máscaras.
- EXCEL_PATH                                        : Ruta del Excel con datos reales.
- OUTPUT_DIR                                        : carpeta base de salida.

Estructura de salida!!!
OUTPUT_DIR/
└── resultados_modelo.csv
"""

import cv2
import torch
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import *
from model import build_model
from dataset import letterbox, unletterbox_mask

# --- FUNCIONES AUXILIARES ---

def red_mask_robust(roi_bgr, tape_mask_u8=None):
    """
    Máscara de rojo robusta (HSV relajado + Lab a-channel).
    Retorna uint8 0/255.
    Si tape_mask_u8 se entrega, enmascara dentro de la huincha.
    """
    h, w = roi_bgr.shape[:2]

    # --- HSV (relajado) ---
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Bajamos S/V mínimos para capturar rojo oscuro en sombra
    s_min = 25
    v_min = 25

    m1 = cv2.inRange(hsv, (0,   s_min, v_min), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, s_min, v_min), (180, 255, 255))

    hsv_mask = cv2.bitwise_or(m1, m2)

    # --- Lab a-channel (más estable en iluminación) ---
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]  # 0..255, alto => más rojo

    # Umbral adaptativo: percentil alto dentro de la huincha si existe
    if tape_mask_u8 is not None and (tape_mask_u8 > 0).any():
        vals = a[tape_mask_u8 > 0]
        thr = int(max(140, np.percentile(vals, 92)))  # ajustable 88..95
    else:
        thr = 150

    lab_mask = (a >= thr).astype(np.uint8) * 255

    # --- Combinar ---
    mask = cv2.bitwise_or(hsv_mask, lab_mask)

    # Enmascarar dentro de la huincha si existe
    if tape_mask_u8 is not None:
        mask[tape_mask_u8 == 0] = 0

    # Limpieza morfológica
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    return mask

def _hsv_red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Rojo en HSV (wrap): [0..10] y [170..180]
    m1 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))
    m = cv2.bitwise_or(m1, m2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m  # uint8 0/255

def _component_has_hole(bin_u8):
    cnts, hier = cv2.findContours(bin_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return False
    hier = hier[0]  # (N,4)
    for i in range(len(cnts)):
        child = hier[i][2]
        parent = hier[i][3]
        if parent == -1 and child != -1:
            return True
    return False

def detect_mark10_from_red_geometry(
    image_bgr,
    huincha_blob,
    margin_px=30,
    min_area=60,
    max_area_ratio=0.25,
    # Heurísticas de posición (en coords normalizadas dentro de la ROI de la huincha)
    right_min_x_norm=0.55,     # 0.55 = buscar en el ~45% derecho
    top_min_y_norm=0.05,       # evitar muy arriba
    bottom_max_y_norm=0.90,    # evitar muy abajo (no cerca del borde inferior)
    target_y_norm=0.55,        # 10 cm suele quedar zona media (ajusta 0.45–0.70)
    target_y_tol=0.35,         # tolerancia del "target_y_norm"
    dilate_iters=1,            # ayuda cuando el rojo está fragmentado
):
    """
    Detecta la marca "10" (decena roja) dentro de la ROI de la huincha usando geometría.
    Robusto a 90°: "1" se detecta por elongación (w/h grande o h/w grande), "0" por agujero (hole).

    Retorna:
      found (bool), y_marca (float), x_marca (float), bbox_xywh (tuple), roi_mask_u8 (uint8 0/255)
    """

    ys, xs = np.where(huincha_blob)
    if xs.size == 0:
        return False, None, None, None, None

    H, W = huincha_blob.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # Expand ROI con margen
    x0 = max(0, x0 - margin_px); x1 = min(W - 1, x1 + margin_px)
    y0 = max(0, y0 - margin_px); y1 = min(H - 1, y1 + margin_px)

    roi = image_bgr[y0:y1 + 1, x0:x1 + 1]
    tape_roi = huincha_blob[y0:y1 + 1, x0:x1 + 1].astype(np.uint8) * 255

    red = red_mask_robust(roi, tape_mask_u8=tape_roi)

    # Opcional: unir fragmentos del rojo
    if dilate_iters and dilate_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red = cv2.dilate(red, k, iterations=int(dilate_iters))
        red[tape_roi == 0] = 0

    if int(red.sum()) < 255 * 30:
        return False, None, None, None, red

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red, connectivity=8)

    roi_h, roi_w = red.shape[:2]
    roi_area = roi_h * roi_w
    max_area = int(max_area_ratio * roi_area)

    zeros = []
    ones = []
    cands = []  # para fallback

    def pos_ok(cx, cy):
        x_norm = cx / max(roi_w, 1)
        y_norm = cy / max(roi_h, 1)
        if x_norm < right_min_x_norm:
            return False
        if y_norm < top_min_y_norm:
            return False
        if y_norm > bottom_max_y_norm:
            return False
        return True

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue

        cx, cy = centroids[i]
        if not pos_ok(cx, cy):
            continue

        elong = max(w, h) / (min(w, h) + 1e-6)   # rotación-invariante
        extent = area / (w * h + 1e-6)

        comp = (labels == i).astype(np.uint8) * 255
        has_hole = _component_has_hole(comp)

        # Guardar candidato para fallback
        cands.append((area, float(cx), float(cy), int(x), int(y), int(w), int(h), bool(has_hole), float(elong), float(extent)))

        # '0' = agujero + no demasiado elongado + razonable en extent
        if has_hole and elong < 3.0 and extent > 0.12:
            zeros.append((i, x, y, w, h, area, float(cx), float(cy)))

        # '1' = elongado (sirve rotado 90°), extent razonable
        if (not has_hole) and elong > 3.0 and extent > 0.10:
            ones.append((i, x, y, w, h, area, float(cx), float(cy)))

    # 1) Intentar emparejar (1,0) más cercano, con penalización por posición
    best = None
    for o in ones:
        for z in zeros:
            # cercanía entre centroides
            dx = o[6] - z[6]
            dy = o[7] - z[7]
            d2 = dx * dx + dy * dy

            # preferir que el centro del par esté cerca de target_y_norm
            cy_pair = 0.5 * (o[7] + z[7])
            y_norm = cy_pair / max(roi_h, 1)
            y_score = 1.0 - min(1.0, abs(y_norm - target_y_norm) / max(target_y_tol, 1e-6))

            # preferir que esté bien a la derecha
            cx_pair = 0.5 * (o[6] + z[6])
            x_norm = cx_pair / max(roi_w, 1)

            # score: minimizar d2, maximizar x_norm y y_score
            score = d2 - 5000.0 * x_norm - 3000.0 * y_score

            if best is None or score < best[0]:
                best = (score, o, z)

    if best is not None:
        _, o, z = best
        x_u = min(o[1], z[1])
        y_u = min(o[2], z[2])
        x2_u = max(o[1] + o[3], z[1] + z[3])
        y2_u = max(o[2] + o[4], z[2] + z[4])

        x_marca = x0 + 0.5 * (x_u + x2_u)
        y_marca = y0 + 0.5 * (y_u + y2_u)
        bbox = (int(x0 + x_u), int(y0 + y_u), int(x2_u - x_u), int(y2_u - y_u))
        return True, float(y_marca), float(x_marca), bbox, red

    # 2) Fallback: si no hay par, escoger el "mejor" blob rojo por posición (derecha) y y cercana a target
    if len(cands) > 0:
        best2 = None
        for area, cx, cy, x, y, w, h, has_hole, elong, extent in cands:
            x_norm = cx / max(roi_w, 1)
            y_norm = cy / max(roi_h, 1)

            y_score = 1.0 - min(1.0, abs(y_norm - target_y_norm) / max(target_y_tol, 1e-6))

            # score: más área + más a la derecha + y cerca de target
            # (ajusta pesos si quieres)
            score = 0.35 * (area / max_area) + 0.40 * x_norm + 0.25 * y_score

            if best2 is None or score > best2[0]:
                best2 = (score, area, cx, cy, x, y, w, h)

        _, area, cx, cy, x, y, w, h = best2
        x_marca = x0 + cx
        y_marca = y0 + cy
        bbox = (int(x0 + x), int(y0 + y), int(w), int(h))
        return True, float(y_marca), float(x_marca), bbox, red

    return False, None, None, None, red

def get_largest_blobs(mask, k=1):
    """
    Extrae los k blobs conexos de mayor área desde una máscara binaria.

    --------------------------------------------------------------------------------

    Parámetros:
    - mask: Máscara binaria de la región de interés (np.ndarray).
    - k: Número de blobs más grandes a retornar (por defecto 1) (int, opcional).

    Retorna:
    - blobs: Lista de máscaras binarias, una por cada blob seleccionado, ordenadas
             por área descendente (list[np.ndarray]).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)

    blobs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        blobs.append((i, area))

    blobs = sorted(blobs, key=lambda x: x[1], reverse=True)[:k]
    return [(labels == idx) for idx, _ in blobs]

def get_all_blobs_with_stats(mask):
    """
    Retorna todos los blobs conexos de una máscara binaria con su área (en px).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    blobs = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        blobs.append((labels == i, area))
    return blobs


def width_of_mask_pca(mask, pct=0, k=1):
    """
    Estima el ancho característico de una máscara binaria mediante PCA.

    --------------------------------------------------------------------------------

    Parámetros:
    - mask: Máscara binaria del objeto de interés (np.ndarray)
    - pct: Percentil a descartar en ambos extremos para robustez (por defecto 0) (float, opcional).
    - k: Número de blobs principales a considerar (por defecto 1) (int, opcional).

    Retorna:
    - width: Ancho estimado en píxeles. Retorna None si no fue posible calcularlo (float or None).
    """
    blobs = get_largest_blobs(mask, k=k)
    if not blobs:
        return None

    pts = []

    for blob in blobs:
        ys, xs = np.where(blob)
        if len(xs) > 0:
            pts.append(np.column_stack([xs, ys]))

    if len(pts) == 0:
        return None

    pts = np.vstack(pts).astype(np.float32)

    # --- PCA ---
    pts -= pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)

    if vt.shape[0] < 2:
        return None

    axis_width = vt[1]
    proj = pts @ axis_width

    width = np.percentile(proj, 100 - pct) - np.percentile(proj, pct)
    return float(width) if width > 0 else None

# --- FUNCIÓN PRINCIPAL ---
def calcular_asentamiento_desde_imagen(image_bgr, model):
    """
    Calcula el asentamiento a partir de una imagen usando segmentación y
    análisis geométrico.

    El procedimiento incluye:
    - Preprocesamiento y redimensionamiento con letterbox.
    - Inferencia con un modelo de segmentación semántica.
    - Identificación de huincha, barra y marca de 10 cm.
    - Cálculo de escala píxel → centímetro.
    - Cálculo del asentamiento en píxeles y centímetros.
    - Extracción de un conjunto extenso de features geométricas y relacionales.

    --------------------------------------------------------------------------------

    Parámetros:
    - image_bgr: Imagen de entrada en formato BGR (np.ndarray).
    - model: Modelo de segmentación cargado en modo evaluación (torch.nn.Module).

    Retorna:
    features: Diccionario con asentamiento estimado y demás features (dict or None).
    """
    # --- Preprocesamiento ---
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_lb, meta = letterbox(image_rgb, IMAGE_SIZE, is_mask=False)

    img_tensor = (
        torch.tensor(image_lb / 255.0, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --- Inferencia ---
    with torch.no_grad():
        logits = model(img_tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    # --- Deshacer letterbox ---
    pred_mask = unletterbox_mask(pred_mask, meta)
    H, W = image_bgr.shape[:2]

    # --- Huincha (nuevo criterio: azul más grande en contacto con rojo más grande) ---

    # 1) Blob rojo más grande (ASENTAMIENTO)
    asent_mask = (pred_mask == CLASE_ASENTAMIENTO)
    asent_blobs = get_largest_blobs(asent_mask, k=1)
    if not asent_blobs:
        return None  # o fallback a tu método antiguo si prefieres
    asent_blob = asent_blobs[0]

    # Dilatar para tolerar pequeños gaps (ajustable)
    kernel = np.ones((3, 3), np.uint8)
    asent_dil = cv2.dilate(asent_blob.astype(np.uint8), kernel, iterations=1).astype(bool)

    # 2) Entre blobs azules (HUINCHA), elegir el más grande que toque al rojo
    huincha_mask = (pred_mask == CLASE_HUINCHA)
    huincha_all = get_all_blobs_with_stats(huincha_mask)

    candidatos = []
    for blob, area in huincha_all:
        # "Contacto" = intersección con el rojo dilatado
        if np.any(blob & asent_dil):
            candidatos.append((blob, area))

    if not candidatos:
        # Fallback: tu método actual (por si hay corte/gap grande)
        huincha_blobs = get_largest_blobs(huincha_mask, k=2)
        if not huincha_blobs:
            return None
        bottom_blob = max(huincha_blobs, key=lambda b: np.max(np.where(b)[0]))
    else:
        # Azul más grande en contacto con rojo
        bottom_blob = max(candidatos, key=lambda t: t[1])[0]

    huincha_width_px = width_of_mask_pca(bottom_blob)       # Ancho "real" de huincha
    if huincha_width_px is None or huincha_width_px <= 0:
        return None

    #cm_per_px = HUINCHA_REAL_ANCHO_CM / huincha_width_px

    ys_h, xs_h = np.where(bottom_blob)
    if len(xs_h) == 0:
        return None

    y_huincha = ys_h.max()      # Posición más baja del blob de huincha

    # --- Marca 10 cm ---
    marca_mask = (pred_mask == CLASE_MARCA_10)
    marca_blobs = get_largest_blobs(marca_mask, k=2)

    kernel = np.ones((3, 3), np.uint8)
    huincha_dil = cv2.dilate(bottom_blob.astype(np.uint8), kernel, iterations=1).astype(bool)

    marca_blob = None
    mark10_source = "seg"
    mark10_bbox = None

    if marca_blobs:
        # Elegir el blob de marca que toque la huincha (entre los 2 más grandes)
        for b in marca_blobs:
            if np.any(b & huincha_dil):
                marca_blob = b
                break
        if marca_blob is None:
            marca_blob = marca_blobs[0]

        ys_m, xs_m = np.where(marca_blob)
        if len(xs_m) > 0:
            y_marca = float(ys_m.mean())
            x_marca = float(xs_m.mean())
            # bbox aproximado del blob segmentado
            x0b, x1b = int(xs_m.min()), int(xs_m.max())
            y0b, y1b = int(ys_m.min()), int(ys_m.max())
            mark10_bbox = (x0b, y0b, x1b - x0b + 1, y1b - y0b + 1)
        else:
            marca_blob = None  # fuerza fallback

    if marca_blob is None:
        # Fallback: detectar "10" rojo dentro de la huincha
        found10, y10, x10, bbox, _ = detect_mark10_from_red_geometry(
            image_bgr,
            bottom_blob,
            margin_px=30,
            min_area=60,
            max_area_ratio=0.25,
        )
        if not found10:
            return None

        y_marca = float(y10)
        x_marca = float(x10)
        mark10_bbox = bbox
        mark10_source = "red"

    # Relación centímetro-píxel
    cm_per_px = 10.0 / max(abs(y_huincha - y_marca), 1e-6)

    # --- Features huincha ---
    huincha_height_bbox_px = ys_h.max() - ys_h.min()        # Altura bbox huincha
    huincha_width_bbox_px = xs_h.max() - xs_h.min()         # Ancho bbox huincha
    huincha_area_px = len(ys_h)         # Área bbox huincha

    huincha_x_mean_px = xs_h.mean()     # Promedio en x del blob de huincha
    huincha_y_mean_px = ys_h.mean()     # Promedio en y del blob de huincha
    huincha_x_min_px = xs_h.min()       # Mínimo en x del blob de huincha
    huincha_x_max_px = xs_h.max()       # Máximo en x del blob de huincha

    huincha_aspect_ratio = huincha_height_bbox_px / (huincha_width_bbox_px + 1e-6)      # Relación de aspecto bbox huincha

    coef_h = np.polyfit(xs_h, ys_h, 1)
    huincha_global_slope = coef_h[0]        # Pendiente global de la huincha

    xs_h_edge = xs_h[ys_h == y_huincha]     # Borde inferior huincha
    huincha_edge_var = np.std(xs_h_edge) if len(xs_h_edge) > 1 else 0.0
    huincha_edge_slope = np.polyfit(np.arange(len(xs_h_edge)), xs_h_edge, 1)[0] if len(xs_h_edge) >= 2 else 0.0     # Pendiente borde inferior huincha
    huincha_edge_ratio = len(xs_h_edge) / len(xs_h)     # Relación borde / total

    # --- Barra ---
    barra_mask = pred_mask == CLASE_BARRA
    barra_blobs = get_largest_blobs(barra_mask, k=2)
    if not barra_blobs:
        return None

    left_blob = min(barra_blobs, key=lambda b: np.mean(np.where(b)[1]))     # Blob de barra más a la izquierda
    ys_b, xs_b = np.where(left_blob)
    if len(xs_b) == 0:
        return None

    # --- Geometría de y_barra ---
    x_ref = np.percentile(xs_b, 98)     # Posición en x del píxel a considerar (percentil 98 para asegurar)

    tolerancia = 2
    ys_b_edge = ys_b[np.abs(xs_b - x_ref) <= tolerancia]    # Banda del borde

    if len(ys_b_edge) == 0:
        return None

    y_barra = ys_b_edge.max()       # Píxel más bajo y más a la derecha del blob de barra izquierda 

    # --- Asentamiento ---
    asentamiento_px = abs(y_huincha - y_barra)          # Asentamiento en px
    asentamiento_cm = asentamiento_px * cm_per_px       # Asentamiento en cm
    distancia_cm_norm_huincha = asentamiento_cm / HUINCHA_REAL_ANCHO_CM     # Asentamiento normalizado por el ancho real de la huincha

    # --- Features barra ---
    barra_height_bbox_px = ys_b.max() - ys_b.min()      # Altura bbox barra
    barra_width_bbox_px = xs_b.max() - xs_b.min()       # Ancho bbox barra
    barra_area_px = len(ys_b)           # Área bbox barra

    barra_x_mean_px = xs_b.mean()       # Promedio en x del blob de barra
    barra_y_mean_px = ys_b.mean()       # Promedio en y del blob de barra
    barra_x_min_px = xs_b.min()         # Mínimo en x del blob de barra
    barra_x_max_px = xs_b.max()         # Máximo en x del blob de barra

    barra_aspect_ratio = barra_height_bbox_px / (barra_width_bbox_px + 1e-6)        # Relación de aspecto bbox barra

    coef_b = np.polyfit(xs_b, ys_b, 1)
    barra_global_slope = coef_b[0]          # Pendiente global de la barra
    barra_edge_var = np.std(ys_b_edge)
    barra_edge_slope = np.polyfit(np.arange(len(ys_b_edge)), ys_b_edge, 1)[0] if len(ys_b_edge) >= 2 else 0.0       # Pendiente del borde derecho de la barra
    barra_edge_ratio = len(ys_b_edge) / len(ys_b)       # Relación borde / total

    # --- Features relacionadas ---
    distancia_norm_barra = asentamiento_px / (barra_height_bbox_px + 1e-6)          # Asentamiento normalizado por la altura de la barra
    distancia_norm_huincha = asentamiento_px / (huincha_height_bbox_px + 1e-6)      # Asentamiento normalizado por la altura de la huincha

    ratio_barra_huincha_height = barra_height_bbox_px / (huincha_height_bbox_px + 1e-6)     # Relación de altura de barra y huincha
    ratio_barra_huincha_area = barra_area_px / (huincha_area_px + 1e-6)         # Relación de áreas de barra y huincha

    slope_diff_barra_huincha = barra_global_slope - huincha_global_slope        # Diferencia de pendientes globales de barra y huincha
    slope_abs_diff_barra_huincha = abs(slope_diff_barra_huincha)            # Diferencia absoluta de pendientes de barra y huincha

    barra_rel_y = barra_y_mean_px / H           # Posición vertical de la barra normalizada
    huincha_rel_y = huincha_y_mean_px / H       # Posición vertical de la huincha normalizada
    delta_rel_y_barra_huincha = barra_rel_y - huincha_rel_y         # Diferencia de posiciones verticales normalizadas de barra y huincha
    delta_x_barra_huincha = (barra_x_mean_px - huincha_x_mean_px) / W       # Diferencia de posiciones horizontales normalizadas de barra y huincha

    barra_edge_y_mean = ys_b_edge.mean()        # Promedio vertical del borde derecho de la barra
    barra_edge_center_offset = (barra_edge_y_mean - barra_y_mean_px) / (barra_height_bbox_px + 1e-6)        # Desplazamiento del borde respecto del centro de la barra (normalizado)

    vertical_slope_penalty_norm = (abs(barra_global_slope) * barra_height_bbox_px) / H      # Penalización normalizada por inclinación vertical de la barra

    geometry_confidence = huincha_edge_ratio * barra_edge_ratio * (1.0 / (1.0 + slope_abs_diff_barra_huincha))      # Confianza geométrica combinada

    angle_abs_diff_barra_huincha = abs(np.arctan(barra_global_slope) - np.arctan(huincha_global_slope))     # Diferencia angular absoluta de barra y huincha
    
    barra_tilt_abs_deg = abs(np.degrees(np.arctan(barra_global_slope)))         # Inclinación de la barra respecto a la vertical
    huincha_vertical_error_deg = abs(90 - abs(np.degrees(np.arctan(huincha_global_slope))))     # Error de verticalidad de la huincha (ideal 90°)

    huincha_y_min_px = ys_h.min()       # Coordenada superior de la huincha
    huincha_y_max_px = ys_h.max()       # Coordenada inferior de la huincha
    
    barra_rel_pos_in_huincha = (barra_y_mean_px - huincha_y_min_px) / (huincha_y_max_px - huincha_y_min_px + 1e-6)  # Posición relativa de la barra dentro de la huincha

    return {
        "x_ref": x_ref, "y_huincha": y_huincha, "y_barra": y_barra, "y_marca": y_marca,

        "asentamiento_modelo_cm": asentamiento_cm, "distancia_px": asentamiento_px,
        "distancia_cm_norm_huincha": distancia_cm_norm_huincha,
        "distancia_norm_barra": distancia_norm_barra, "distancia_norm_huincha": distancia_norm_huincha,

        "delta_rel_y_barra_huincha": delta_rel_y_barra_huincha,
        "delta_x_barra_huincha": delta_x_barra_huincha,
        "ratio_barra_huincha_height": ratio_barra_huincha_height,
        "ratio_barra_huincha_area": ratio_barra_huincha_area,
        "barra_rel_pos_in_huincha": barra_rel_pos_in_huincha,

        "barra_rel_y": barra_rel_y, "huincha_rel_y": huincha_rel_y,

        "barra_height_px": barra_height_bbox_px, "barra_width_px": barra_width_bbox_px,
        "barra_area_px": barra_area_px, "barra_aspect_ratio": barra_aspect_ratio,
        "barra_tilt_abs_deg": barra_tilt_abs_deg,
        "barra_x_mean_px": barra_x_mean_px, "barra_y_mean_px": barra_y_mean_px,
        "barra_x_min_px": barra_x_min_px, "barra_x_max_px": barra_x_max_px,

        "huincha_height_px": huincha_height_bbox_px, "huincha_width_px": huincha_width_px,
        "huincha_area_px": huincha_area_px, "huincha_aspect_ratio": huincha_aspect_ratio,
        "huincha_vertical_error_deg": huincha_vertical_error_deg,
        "huincha_x_mean_px": huincha_x_mean_px, "huincha_y_mean_px": huincha_y_mean_px,
        "huincha_x_min_px": huincha_x_min_px, "huincha_x_max_px": huincha_x_max_px,

        "barra_global_slope": barra_global_slope, "barra_edge_slope": barra_edge_slope,
        "huincha_global_slope": huincha_global_slope, "huincha_edge_slope": huincha_edge_slope,
        "slope_diff_barra_huincha": slope_diff_barra_huincha,
        "slope_abs_diff_barra_huincha": slope_abs_diff_barra_huincha,
        "angle_abs_diff_barra_huincha": angle_abs_diff_barra_huincha,

        "barra_edge_var": barra_edge_var, "barra_edge_ratio": barra_edge_ratio,
        "barra_edge_center_offset": barra_edge_center_offset,
        "huincha_edge_var": huincha_edge_var, "huincha_edge_ratio": huincha_edge_ratio,
        "vertical_slope_penalty_norm": vertical_slope_penalty_norm,
        "geometry_confidence": geometry_confidence,

        "mark10_source": mark10_source,
        "mark10_bbox_x": mark10_bbox[0] if mark10_bbox else np.nan,
        "mark10_bbox_y": mark10_bbox[1] if mark10_bbox else np.nan,
        "mark10_bbox_w": mark10_bbox[2] if mark10_bbox else np.nan,
        "mark10_bbox_h": mark10_bbox[3] if mark10_bbox else np.nan,
    }


# --- MAIN ---
def main():
    """
    Ejecuta el procedimiento completo de evaluación y corrección del asentamiento.

    Flujo general:
    - Carga del modelo de segmentación entrenado.
    - Inferencia sobre imágenes de train, val y test.
    - Extracción de features geométricas y cálculo del asentamiento.
    - Comparación con valores reales obtenidos desde el archivo Excel.
    - Cálculo de métricas base (MAE, RMSE, sesgo).
    - Entrenamiento de una regresión correctora usando solo TRAIN.
    - Aplicación de la corrección a todos los conjuntos.
    - Evaluación final y guardado de resultados en formato CSV.
    """
    # --- Modelo ---
    model = build_model(encoder=ENCODER_NAME, pesos=ENCODER_WEIGHTS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("Modelo cargado correctamente\n")

    resultados = []

    SPLITS = {
        "train": TRAIN_IMG_DIR,
        "val": VAL_IMG_DIR,
        "test": TEST_IMG_DIR,
    }

    # --- Inferencia + extracción de features ---
    for split, img_dir in SPLITS.items():
        print(f"Procesando split: {split}")

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in EXTS:
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            out = calcular_asentamiento_desde_imagen(image, model)
            if out is None:
                continue

            resultados.append({
                "N° muestra": img_path.stem,
                "split": split,
                **out,
            })

    df_modelo = pd.DataFrame(resultados)

    # --- Datos reales ---
    df_real = pd.read_excel(EXCEL_PATH)
    df_real = df_real.rename(
        columns={"N° Asentamiento (cm)": "asentamiento_real_cm"}
    )
    df_real["N° muestra"] = df_real["N° muestra"].astype(str)

    df = pd.merge(df_real, df_modelo, on="N° muestra", how="inner")
    print(f"\nMuestras comparadas: {len(df)}")

    # --- Métricas base ---
    df["error_cm"] = df["asentamiento_modelo_cm"] - df["asentamiento_real_cm"]

    print("\nMÉTRICAS BASE:")
    print(f"MAE:   {df['error_cm'].abs().mean():.2f} cm")
    print(f"RMSE:  {np.sqrt((df['error_cm']**2).mean()):.2f} cm")
    print(f"Sesgo: {df['error_cm'].mean():.2f} cm")

    base_by_split = (
        df
        .groupby("split")
        .agg(
            MAE=("error_cm", lambda x: x.abs().mean()),
            RMSE=("error_cm", lambda x: np.sqrt((x**2).mean())),
            Sesgo=("error_cm", "mean"),
        )
    )

    print("\nMÉTRICAS BASE POR CONJUNTO:")
    print(base_by_split.round(2))

    # --- Regresión correctora ---
    features = [
        "y_huincha",
        "y_barra",
        "y_marca",
        "x_ref",

        "asentamiento_modelo_cm",
        "distancia_cm_norm_huincha",

        "distancia_px",
        "distancia_norm_barra",
        "distancia_norm_huincha",
        "delta_rel_y_barra_huincha",
        "delta_x_barra_huincha",

        "ratio_barra_huincha_height",
        "ratio_barra_huincha_area",

        "slope_diff_barra_huincha",
        "slope_abs_diff_barra_huincha",
        "angle_abs_diff_barra_huincha",

        "barra_rel_pos_in_huincha",

        "barra_rel_y",
        "huincha_rel_y",

        "barra_height_px",
        "barra_width_px",
        "barra_area_px",
        "barra_aspect_ratio",
        "barra_tilt_abs_deg",

        "barra_x_mean_px",
        "barra_y_mean_px",
        "barra_x_min_px",
        "barra_x_max_px",

        "huincha_height_px",
        "huincha_width_px",
        "huincha_area_px",
        "huincha_aspect_ratio",
        "huincha_vertical_error_deg",

        "huincha_x_mean_px",
        "huincha_y_mean_px",
        "huincha_x_min_px",
        "huincha_x_max_px",

        "barra_global_slope",
        "barra_edge_slope",
        "huincha_global_slope",
        "huincha_edge_slope",

        "barra_edge_var",
        "barra_edge_ratio",
        "barra_edge_center_offset",

        "huincha_edge_var",
        "huincha_edge_ratio",

        "vertical_slope_penalty_norm",
        "geometry_confidence",
    ]

    # Regresión solo con train
    df_train = df[df["split"] == "train"]
    if len(df_train) == 0:
        raise ValueError("No hay muestras en TRAIN para entrenar la regresión")

    X_train = df_train[features]
    y_train = df_train["error_cm"]

    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    reg.fit(X_train, y_train)
    print("\nRegresión correctora entrenada (solo TRAIN)")

    # Aplicar a todos los splits
    error_pred = reg.predict(df[features])

    df["asentamiento_modelo_corr_cm"] = (
        df["asentamiento_modelo_cm"] - error_pred
    )
    df["error_corr_cm"] = (
        df["asentamiento_modelo_corr_cm"] - df["asentamiento_real_cm"]
    )

    # Métricas finales
    print("\nMÉTRICAS FINALES (CON REGRESIÓN):")
    print(f"MAE:   {df['error_corr_cm'].abs().mean():.2f} cm")
    print(f"RMSE:  {np.sqrt((df['error_corr_cm']**2).mean()):.2f} cm")
    print(f"Sesgo: {df['error_corr_cm'].mean():.2f} cm")

    final_by_split = (
        df
        .groupby("split")
        .agg(
            MAE=("error_corr_cm", lambda x: x.abs().mean()),
            RMSE=("error_corr_cm", lambda x: np.sqrt((x**2).mean())),
            Sesgo=("error_corr_cm", "mean"),
        )
    )

    print("\nMÉTRICAS FINALES POR CONJUNTO:")
    print(final_by_split.round(2))

    # --- Guardar resultados para calc_viz.py ---
    RESULTS_PATH = OUTPUT_DIR / "resultados_modelo.csv"
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(RESULTS_PATH, index=False)

    print(f"\nResultados guardados en: {RESULTS_PATH}")

if __name__ == "__main__":
    main()