# app.py

from __future__ import annotations

from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
import cv2
import torch
import streamlit as st
from joblib import load as joblib_load

import config as cfg
from model import build_model
from dataset import letterbox, unletterbox_mask
from utils import slump_calc2


# -----------------------------
# Rutas fijas
# -----------------------------
ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "outputs" / "modelo_conos_10cm.pth"
REGRESSION_PATH = ROOT / "outputs" / "Resultados 2 - Huincha y marca 10cm" / "regresion_correctora.joblib"


# -----------------------------
# Visual helpers (calc_viz style)
# -----------------------------
def _invert_color_map(color_map: dict) -> dict:
    return {v: k for k, v in color_map.items()}  # class_id -> (R,G,B)


def class_to_color(mask_hw: np.ndarray, inv_color_map: dict) -> np.ndarray:
    h, w = mask_hw.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in inv_color_map.items():
        color_mask[mask_hw == cls] = color
    return color_mask


def overlay(image_rgb: np.ndarray, mask_color_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(image_rgb, 1 - alpha, mask_color_rgb, alpha, 0)


def put_text_with_bg(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font,
    font_scale: float,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness: int = 2,
    padding: int = 6,
):
    (x, y) = org
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        img,
        (x - padding, y - h - padding),
        (x + w + padding, y + baseline + padding),
        bg_color,
        thickness=-1,
    )
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)


def encode_png_bytes(rgb_img: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("No se pudo codificar PNG.")
    return buf.tobytes()


# -----------------------------
# Device automático (sin UI)
# -----------------------------
def auto_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Persistencia de resultados entre reruns
# ------------------------------
def uploads_signature(uploads) -> tuple | None:
    if not uploads:
        return None
    # firma barata: nombre + tamaño
    sig = []
    for up in uploads:
        size = getattr(up, "size", None)
        if size is None:
            size = len(up.getvalue())
        sig.append((up.name, int(size)))
    return tuple(sig)

if "processed_sig" not in st.session_state:
    st.session_state["processed_sig"] = None

if "results_cache" not in st.session_state:
    st.session_state["results_cache"] = None


# -----------------------------
# Carga de modelo / regresión (cache)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_segmentation_model(weights_path: Path, device: str) -> torch.nn.Module:
    try:
        cfg.DEVICE = device
    except Exception:
        pass

    model = build_model(encoder=cfg.ENCODER_NAME, pesos=cfg.ENCODER_WEIGHTS)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_regression(joblib_path: Path):
    payload = joblib_load(joblib_path)
    if isinstance(payload, dict) and "model" in payload:
        reg = payload["model"]
        feature_cols = payload.get("feature_cols", None)
    else:
        reg = payload
        feature_cols = None
    return reg, feature_cols


# -----------------------------
# Inferencia máscara (overlay)
# -----------------------------
@torch.no_grad()
def predict_mask_classes(image_bgr: np.ndarray, model: torch.nn.Module, device: str) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_lb, meta = letterbox(image_rgb, cfg.IMAGE_SIZE, is_mask=False)

    img_tensor = (
        torch.tensor(image_lb / 255.0, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    logits = model(img_tensor)
    pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    pred_mask = unletterbox_mask(pred_mask, meta)
    return pred_mask


def run_one(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
    device: str,
    inv_color_map: dict,
    do_overlay: bool,
    alpha_overlay: float,
    draw_lines: bool,
    font_scale: float,
    reg=None,
    feature_cols=None,
):
    feats = slump_calc2.calcular_asentamiento_desde_imagen(image_bgr, model)
    if feats is None:
        raise RuntimeError("No se pudo calcular asentamiento (feats=None).")

    asent_modelo = float(feats["asentamiento_modelo_cm"])

    error_pred = None
    asent_corregido = None
    if reg is not None and feature_cols:
        row = {k: feats.get(k, 0.0) for k in feature_cols}
        X = pd.DataFrame([row], columns=feature_cols)
        for c in feature_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

        error_pred = float(reg.predict(X)[0])
        asent_corregido = asent_modelo - error_pred
        if asent_corregido < 0:
            asent_corregido = 0.0

    vis = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if do_overlay:
        pred_mask = predict_mask_classes(image_bgr, model, device)
        pred_color = class_to_color(pred_mask, inv_color_map)
        vis = overlay(vis, pred_color, alpha=alpha_overlay)

    h, w = vis.shape[:2]
    if draw_lines:
        y_huincha = int(feats.get("y_huincha", -1))
        y_barra = int(feats.get("y_barra", -1))
        y_marca = int(feats.get("y_marca", -1))
        x_ref = int(feats.get("x_ref", -1))

        if 0 <= y_huincha < h:
            cv2.line(vis, (0, y_huincha), (w - 1, y_huincha), (255, 255, 0), 2)
        if 0 <= y_barra < h:
            cv2.line(vis, (0, y_barra), (w - 1, y_barra), (255, 0, 0), 2)
        if 0 <= y_marca < h:
            cv2.line(vis, (0, y_marca), (w - 1, y_marca), (0, 0, 255), 2)
        if 0 <= x_ref < w:
            cv2.line(vis, (x_ref, 0), (x_ref, h - 1), (255, 0, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    put_text_with_bg(
        vis, f"Modelo: {asent_modelo:.2f} cm", (10, 40), font, font_scale,
        text_color=(249, 250, 155), bg_color=(0, 0, 0)
    )
    if asent_corregido is not None:
        put_text_with_bg(
            vis, f"Corregido: {asent_corregido:.2f} cm", (10, 85), font, font_scale,
            text_color=(255, 119, 119), bg_color=(0, 0, 0)
        )

    return vis, feats, asent_modelo, error_pred, asent_corregido


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Asentamiento", layout="wide")
st.title("Cálculo de asentamiento desde imágenes")

# Sidebar: uploader + controles
with st.sidebar:
    st.header("Carga y opciones")

    uploads = st.file_uploader(
        "Arrastra y suelta imágenes aquí",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True
    )

    current_sig = uploads_signature(uploads)

    # Si el usuario cargó imágenes nuevas (o quitó las existentes), borrar resultados anteriores
    if current_sig != st.session_state["processed_sig"]:
        st.session_state["results_cache"] = None

    st.divider()
    do_overlay = st.checkbox("Overlay máscara", value=True)
    alpha_overlay = st.slider("Alpha overlay", 0.0, 1.0, 0.5, 0.05)
    draw_lines = st.checkbox("Dibujar líneas", value=True)
    font_scale = st.slider("Tamaño texto", 0.5, 2.0, 1.2, 0.1)

    st.divider()
    process_btn = st.button("Procesar", type="primary", use_container_width=True)

# Cuerpo: descripción + instrucciones + resultados
st.markdown(
    """
### ¿Qué hace esta aplicación?

Esta app estima el **asentamiento** a partir de una imagen del ensayo.
1) Un modelo de segmentación identifica elementos clave (p. ej., huincha/barra).
2) Con esa geometría se calcula un asentamiento **“Modelo”**.
3) Luego, se aplica una **regresión correctora** para entregar también un asentamiento **“Corregido”**.

### Instrucciones rápidas
1) En la barra lateral, **carga** una o más imágenes (drag & drop).
2) Activa/desactiva **Overlay** y **Líneas** según necesites.
3) Presiona **Procesar**.
4) Descarga resultados como **PNG**, **CSV** o un **ZIP** con todo.
"""
)

st.info(
    "Sugerencias: usa imágenes nítidas, con buena iluminación y donde se vea claramente la referencia de medición."
)

if uploads:
    st.write(f"Imágenes cargadas: **{len(uploads)}**")

if process_btn:
    if not uploads:
        st.error("No cargaste imágenes.")
        st.stop()

    if not WEIGHTS_PATH.exists():
        st.error("No se encontró el archivo de pesos del modelo en el servidor.")
        st.stop()

    device = auto_device()

    with st.spinner("Cargando modelo..."):
        model = load_segmentation_model(WEIGHTS_PATH, device)

    reg = None
    feature_cols = None
    if REGRESSION_PATH.exists():
        try:
            reg, feature_cols = load_regression(REGRESSION_PATH)
            if not feature_cols:
                reg, feature_cols = None, None
        except Exception:
            reg, feature_cols = None, None

    inv_color_map = _invert_color_map(cfg.COLOR_MAP)

    resultados = []
    images_out = []

    zip_buf = io.BytesIO()
    zf = zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED)

    prog = st.progress(0)

    for i, up in enumerate(uploads):
        file_bytes = np.frombuffer(up.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        try:
            vis_rgb, feats, asent_modelo, error_pred, asent_corregido = run_one(
                img_bgr, model, device, inv_color_map,
                do_overlay=do_overlay,
                alpha_overlay=float(alpha_overlay),
                draw_lines=draw_lines,
                font_scale=float(font_scale),
                reg=reg,
                feature_cols=feature_cols,
            )
        except Exception:
            vis_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            put_text_with_bg(
                vis_rgb, "Fallo: no se pudo calcular asentamiento", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), (0, 0, 0)
            )
            asent_modelo, error_pred, asent_corregido = np.nan, np.nan, np.nan

        png_bytes = encode_png_bytes(vis_rgb)
        stem = Path(up.name).stem

        # Guardar imagen en ZIP y también en cache para re-render
        zf.writestr(f"images/{stem}.png", png_bytes)
        images_out.append({"name": up.name, "stem": stem, "png": png_bytes})

        resultados.append({
            "Nombre": up.name,
            "Asentamiento [cm]": asent_modelo,
            "Corrección [cm]": error_pred,
            "Asentamiento corregido [cm]": asent_corregido,
        })

        prog.progress(int((i + 1) / len(uploads) * 100))

    prog.empty()

    df = pd.DataFrame(resultados)

    # Tabla: 2 decimales
    df_show = df.copy()
    num_cols = ["Asentamiento [cm]", "Corrección [cm]", "Asentamiento corregido [cm]"]
    for c in num_cols:
        df_show[c] = pd.to_numeric(df_show[c], errors="coerce")
        df_show[c] = df_show[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    # CSV (también 2 decimales)
    csv_bytes = df.to_csv(index=False, float_format="%.2f").encode("utf-8")

    zf.writestr("inferencia_resultados.csv", csv_bytes)
    zf.close()
    zip_bytes = zip_buf.getvalue()

    # Guardar cache para que no se pierda al apretar Download
    st.session_state["processed_sig"] = current_sig
    st.session_state["results_cache"] = {
        "images": images_out,
        "df_show": df_show,
        "csv_bytes": csv_bytes,
        "zip_bytes": zip_bytes,
    }

# -----------------------------
# Mostrar resultados desde cache (persiste entre reruns)
# -----------------------------
cache = st.session_state.get("results_cache")
if cache:
    cols = st.columns(2)
    for i, item in enumerate(cache["images"]):
        with cols[i % 2]:
            st.image(item["png"], caption=item["name"], use_container_width=True)
            st.download_button(
                "Descargar PNG",
                data=item["png"],
                file_name=f"{item['stem']}.png",
                mime="image/png",
                key=f"dl_img_{i}",
            )

    st.divider()
    st.subheader("Resultados")
    st.dataframe(cache["df_show"], use_container_width=False, width=900, hide_index=True)

    st.download_button(
        "Descargar CSV",
        data=cache["csv_bytes"],
        file_name="inferencia_resultados.csv",
        mime="text/csv",
        key="dl_csv",
    )

    st.download_button(
        "Descargar ZIP (imágenes + CSV)",
        data=cache["zip_bytes"],
        file_name="inferencia_salidas.zip",
        mime="application/zip",
        key="dl_zip",
    )
