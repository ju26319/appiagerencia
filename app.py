"""
Sistema de Inspección de Lotes por Muestreo – Distribuidora ANCO S.A.S.
ISO 2859-1 · YOLOv8n (cuerpo) + Claude Vision (etiqueta) · Gerencia y Control de Calidad
"""

import streamlit as st
import anthropic
import base64
import io
import os
import json
import re
import time
import math
import tempfile
# cv2 y numpy se importan dentro de las funciones YOLO para evitar errores en entornos headless
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from collections import Counter, defaultdict
from datetime import datetime

# Ultralytics y cv2 importados con manejo de error para entornos headless/sin GPU
try:
    import numpy as np
    import cv2
    from ultralytics import YOLO
    YOLO_DISPONIBLE = True
except Exception:
    YOLO_DISPONIBLE = False
    np = None
    cv2 = None
    YOLO = None

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ANCO – Inspección de Lotes",
    page_icon="🥫",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS  (industrial / utilitarian – azul oscuro + ámbar)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #0b1120;
    color: #c8d6e5;
    font-family: 'IBM Plex Sans', sans-serif;
}
section[data-testid='stSidebar'] {
    background: #080e1a;
    border-right: 1px solid #1a2744;
}
.stTabs [data-baseweb="tab-list"] {
    background: #0d1830;
    border-bottom: 2px solid #1a2744;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #5a7a9a !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 10px 20px;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #f0b429 !important;
    border-bottom: 2px solid #f0b429 !important;
    background: transparent !important;
}

/* Cabecera hero */
.hero {
    background: linear-gradient(135deg, #0d1f40 0%, #0b1120 100%);
    border: 1px solid #1a3a6a;
    border-left: 4px solid #f0b429;
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #f0b429;
    margin: 0 0 4px 0;
    letter-spacing: 1px;
}
.hero-sub {
    font-size: 0.85rem;
    color: #5a7a9a;
    margin: 0;
    letter-spacing: 0.5px;
}

/* Tarjetas de métricas */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 16px 0;
}
.met {
    background: #0d1830;
    border: 1px solid #1a2744;
    border-top: 3px solid;
    border-radius: 4px;
    padding: 14px;
    text-align: center;
}
.met.ok   { border-top-color: #27c97e; }
.met.warn { border-top-color: #f0b429; }
.met.bad  { border-top-color: #e55353; }
.met.neu  { border-top-color: #3a7bd5; }
.met-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    display: block;
}
.met-ok   { color: #27c97e; }
.met-warn { color: #f0b429; }
.met-bad  { color: #e55353; }
.met-neu  { color: #3a7bd5; }
.met-lbl  { font-size: 10px; color: #4a6a8a; letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }

/* Tarjeta de lata */
.lata-card {
    background: #0d1830;
    border: 1px solid #1a2744;
    border-radius: 4px;
    padding: 14px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.lata-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    color: #7aa3cc;
    min-width: 60px;
}
.lata-detail { flex: 1; font-size: 13px; color: #7a9ab8; }
.badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 2px;
    letter-spacing: 1px;
}
.badge-ok   { background: #0a2a1a; color: #27c97e; border: 1px solid #1a5a3a; }
.badge-nc   { background: #2a0a0a; color: #e55353; border: 1px solid #5a1a1a; }
.badge-obs  { background: #2a200a; color: #f0b429; border: 1px solid #5a4010; }

/* Decisión de lote */
.decision-box {
    border-radius: 6px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.decision-acepta {
    background: #071a0f;
    border: 2px solid #27c97e;
}
.decision-rechaza {
    background: #1a0707;
    border: 2px solid #e55353;
}
.decision-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    letter-spacing: 3px;
}
.decision-sub { font-size: 13px; color: #7a9ab8; margin-top: 8px; }

/* Historial sidebar */
.hist-item {
    background: #0d1830;
    border: 1px solid #1a2744;
    border-radius: 3px;
    padding: 8px 10px;
    margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.hist-id   { color: #7aa3cc; }
.hist-stat { }

/* Plan info */
.plan-box {
    background: #0d1830;
    border: 1px solid #1a2744;
    border-left: 3px solid #3a7bd5;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    margin: 12px 0;
}
.plan-row { display: flex; gap: 24px; flex-wrap: wrap; }
.plan-param { color: #3a7bd5; }
.plan-val   { color: #f0b429; font-weight: 600; }

.stButton > button {
    background: #0d1830;
    border: 1px solid #1a3a6a;
    color: #c8d6e5;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    border-radius: 3px;
    transition: all 0.15s;
}
.stButton > button:hover {
    border-color: #f0b429;
    color: #f0b429;
    background: #1a2a10;
}
.footer {
    text-align: center;
    color: #1a3a5a;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #1a2744;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABLAS ISO 2859-1  (Nivel de Inspección General II · NCA 2.5%)
# ══════════════════════════════════════════════════════════════════════════════
ISO_LETRA = [
    (2,    8,    "A"),
    (9,    15,   "B"),
    (16,   25,   "C"),
    (26,   50,   "D"),
    (51,   90,   "E"),
    (91,   150,  "F"),
    (151,  280,  "G"),
    (281,  500,  "H"),
    (501,  1200, "J"),
    (1201, 3200, "K"),
    (3201, 10000,"L"),
]

# Tabla II-A: letra → (n, c) para NCA = 2.5%
ISO_PLAN = {
    "A": (2,   0),
    "B": (3,   0),
    "C": (5,   0),
    "D": (8,   0),
    "E": (13,  1),
    "F": (20,  1),
    "G": (32,  2),
    "H": (50,  3),
    "J": (80,  5),
    "K": (125, 7),
    "L": (200, 10),
}

def iso_letra(N):
    for lo, hi, letra in ISO_LETRA:
        if lo <= N <= hi:
            return letra
    return "L"

def iso_plan(N):
    letra = iso_letra(N)
    n, c = ISO_PLAN[letra]
    return letra, min(n, N), c

# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULO DE CCO  (distribución binomial)
# ══════════════════════════════════════════════════════════════════════════════
def comb(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def pa_binomial(p, n, c):
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    total = 0.0
    for x in range(c + 1):
        total += comb(n, x) * (p ** x) * ((1 - p) ** (n - x))
    return min(max(total, 0.0), 1.0)

def calcular_cco(n, c, puntos=60):
    ps = [i / puntos for i in range(puntos + 1)]
    pas = [pa_binomial(p, n, c) for p in ps]
    return ps, pas

def encontrar_riesgos(n, c, nca=0.025, beta_obj=0.10):
    """Devuelve alpha en NCA y NQL (punto donde Pa ≤ beta_obj)."""
    pa_nca = pa_binomial(nca, n, c)
    alpha = 1 - pa_nca
    nql = None
    for i in range(1, 101):
        p = i / 100
        if pa_binomial(p, n, c) <= beta_obj:
            nql = p
            break
    return alpha, nql

# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS PARA CLAUDE VISION – alta precisión
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# YOLO – MODELO DE DEFECTOS FÍSICOS (best_latas_defectos.pt)
# ══════════════════════════════════════════════════════════════════════════════

# Mapeo de clases del modelo entrenado (según dataset Canned Food Surface Defect)
YOLO_CLASES = {
    "Critical Defect": "CRITICO",
    "Major Defect":    "MAYOR",
    "Minor Defect":    "MENOR",
    "No defect":       "CONFORME",
}

YOLO_MODEL_PATH = "best_latas_defectos.pt"
YOLO_CONF_THR   = 0.50   # Recomendado en la ficha técnica del modelo
YOLO_IMGSZ      = 416    # Input shape del modelo entrenado

@st.cache_resource
def cargar_yolo():
    """Carga el modelo YOLO una sola vez y lo cachea en memoria."""
    if not YOLO_DISPONIBLE or YOLO is None:
        return None
    if not os.path.exists(YOLO_MODEL_PATH):
        return None
    try:
        model = YOLO(YOLO_MODEL_PATH)
        return model
    except Exception:
        return None

def analizar_cuerpo_yolo(model, img: Image.Image) -> dict:
    """
    Analiza el cuerpo de la lata con YOLOv8n.
    Retorna la detección de mayor confianza.
    Si no detecta nada → CONFORME (per ficha técnica: ausencia = sin defecto).
    conf_thr=0.50 según nota del desarrollador en ficha técnica.
    """
    if not YOLO_DISPONIBLE or cv2 is None or np is None:
        return {"clase": "ERROR", "confianza": 0.0, "descripcion": "YOLO/cv2 no disponible en este entorno", "boxes": []}
    img_np = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    try:
        results = model(img_np, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_THR, verbose=False)
    except Exception as e:
        return {"clase": "ERROR", "confianza": 0.0, "descripcion": f"Error YOLO: {e}", "boxes": []}

    detecciones = []
    for r in results:
        for box in r.boxes:
            class_id   = int(box.cls[0])
            class_name = model.names[class_id]
            conf       = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            clase_norm = YOLO_CLASES.get(class_name, "CONFORME")
            detecciones.append({
                "clase_raw": class_name,
                "clase": clase_norm,
                "confianza": round(conf, 3),
                "box": (x1, y1, x2, y2),
            })

    if not detecciones:
        # Sin detecciones = conforme (per ficha técnica)
        return {
            "clase": "CONFORME",
            "confianza": 0.92,
            "descripcion": "Sin defectos detectados por YOLO",
            "boxes": [],
            "fuente": "YOLO",
        }

    # Tomar la detección de mayor confianza que no sea CONFORME
    # Si todas son CONFORME, queda CONFORME
    priority = {"CRITICO": 4, "MAYOR": 3, "MENOR": 2, "CONFORME": 1, "ERROR": 0}
    detecciones.sort(key=lambda d: (priority.get(d["clase"], 0), d["confianza"]), reverse=True)
    mejor = detecciones[0]

    # Descripción automática
    desc_map = {
        "CRITICO": f"Defecto crítico detectado ({mejor['clase_raw']}) con {mejor['confianza']:.0%} de confianza",
        "MAYOR":   f"Defecto mayor detectado ({mejor['clase_raw']}) con {mejor['confianza']:.0%} de confianza",
        "MENOR":   f"Defecto menor detectado ({mejor['clase_raw']}) con {mejor['confianza']:.0%} de confianza",
        "CONFORME": f"Lata conforme según YOLO con {mejor['confianza']:.0%} de confianza",
    }

    return {
        "clase": mejor["clase"],
        "confianza": mejor["confianza"],
        "descripcion": desc_map.get(mejor["clase"], ""),
        "boxes": [(d["box"], d["clase"], d["confianza"]) for d in detecciones],
        "fuente": "YOLO",
    }

def dibujar_boxes(img: Image.Image, boxes: list) -> Image.Image:
    """Dibuja bounding boxes del YOLO sobre la imagen para visualización."""
    img_out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img_out)
    colores = {"CRITICO": "#e55353", "MAYOR": "#f0b429", "MENOR": "#3a7bd5", "CONFORME": "#27c97e"}
    for (x1, y1, x2, y2), clase, conf in boxes:
        color = colores.get(clase, "#ffffff")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, y1 + 4), f"{clase} {conf:.0%}", fill=color)
    return img_out

PROMPT_ETIQUETA = """Eres un inspector de control de calidad especializado en etiquetas y fechas de vencimiento de conservas enlatadas.
Analiza esta imagen de la ETIQUETA y determina el estado de la fecha de vencimiento.

Busca:
1. Fecha de vencimiento (también puede aparecer como: VENCE, VENC, EXP, BEST BEFORE, BB, CONSUME ANTES DE, CAD, FECHA LÍMITE).
2. El formato puede ser: DD/MM/AAAA, MM/AAAA, DD-MM-AA, AAAA-MM-DD, o texto impreso en el borde/fondo de la lata.
3. Si hay múltiples fechas, usa la de vencimiento (no la de fabricación/LOT/LOTE).

Reglas de validación (fecha de hoy: """ + datetime.now().strftime("%d/%m/%Y") + """):
- VIGENTE: fecha encontrada, formato válido, y la fecha es posterior a hoy.
- VENCIDA: fecha encontrada y es anterior o igual a hoy.
- ILEGIBLE: hay indicios de que hay una fecha pero no se puede leer con certeza.
- SIN_FECHA: no se encuentra ningún dato de fecha en la imagen.

Formato incorrecto: si el texto existe pero no puede interpretarse como fecha válida → ILEGIBLE.

Ante la duda en si está vigente o vencida, marca VENCIDA (principio de precaución).
La confianza debe ser ≥ 0.85 si la fecha es claramente visible, < 0.70 si hay dudas de lectura.

Responde SOLO con este JSON, sin texto adicional, sin markdown:
{"estado":"VIGENTE|VENCIDA|ILEGIBLE|SIN_FECHA","fecha_leida":"DD/MM/AAAA o texto encontrado o null","confianza":0.00,"descripcion":"descripcion breve en español"}"""

PROMPT_CLAUDE_CUERPO_FALLBACK = """Eres un inspector de control de calidad de conservas enlatadas.
Analiza esta imagen del CUERPO de la lata. Solo si YOLO no está disponible.
- CRITICO: abombamiento, perforación, corrosión profunda, fuga visible.
- MAYOR: abolladuras en sello, deformación estructural, oxidación extendida.
- MENOR: abolladuras leves en zona central, raspones superficiales.
- CONFORME: sin defectos visibles.
Responde SOLO con este JSON sin texto adicional:
{"clase":"CRITICO|MAYOR|MENOR|CONFORME","confianza":0.00,"descripcion":"breve descripcion en español"}"""

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE VISIÓN
# ══════════════════════════════════════════════════════════════════════════════
MODELOS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

def pil_a_b64(img: Image.Image, calidad: int = 92) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=calidad)
    return base64.standard_b64encode(buf.getvalue()).decode()

def preprocesar(img: Image.Image, min_dim: int = 640) -> Image.Image:
    """Mejora contraste y nitidez antes de enviar a Claude."""
    w, h = img.size
    if min(w, h) < min_dim:
        factor = min_dim / min(w, h)
        img = img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(1.8)
    return img

def parsear_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"```[a-z]*", "", text).strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None

def llamar_claude(client, b64: str, prompt: str, modelo: str) -> dict | None:
    resp = client.messages.create(
        model=modelo,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return parsear_json(resp.content[0].text)

def analizar_con_consenso(client, img: Image.Image, prompt: str, modelo: str, intentos: int = 3) -> dict:
    """
    Triple votación: llama 3 veces y toma la clase más frecuente.
    Promedia las confianzas de las respuestas ganadoras.
    Esto eleva la precisión al eliminar variabilidad estocástica.
    """
    img = preprocesar(img)
    b64 = pil_a_b64(img)
    resultados = []

    for _ in range(intentos):
        try:
            r = llamar_claude(client, b64, prompt, modelo)
            if r:
                resultados.append(r)
        except anthropic.NotFoundError:
            break
        except Exception:
            pass

    if not resultados:
        return {"clase": "ERROR", "estado": "ERROR", "confianza": 0.0, "descripcion": "Sin respuesta del modelo"}

    # Determinar campo clave (cuerpo → "clase", etiqueta → "estado")
    campo = "clase" if "clase" in resultados[0] else "estado"
    votos = Counter(r.get(campo, "ERROR") for r in resultados)
    ganador = votos.most_common(1)[0][0]

    # Confianza promedio de las respuestas con el valor ganador
    confs = [r.get("confianza", 0.0) for r in resultados if r.get(campo) == ganador]
    conf_final = sum(confs) / len(confs) if confs else 0.0

    # Descripción de la primera respuesta ganadora
    desc = next((r.get("descripcion", "") for r in resultados if r.get(campo) == ganador), "")

    base = {campo: ganador, "confianza": round(conf_final, 3), "descripcion": desc}

    # Añadir fecha_leida si viene de etiqueta
    if campo == "estado":
        fechas = [r.get("fecha_leida") for r in resultados if r.get(campo) == ganador and r.get("fecha_leida")]
        base["fecha_leida"] = fechas[0] if fechas else None

    return base

def es_no_conforme(res_cuerpo: dict, res_etiqueta: dict) -> tuple[bool, str]:
    """Regla de decisión por lata según ISO y política ANCO."""
    motivos = []
    clase = res_cuerpo.get("clase", "ERROR")
    estado = res_etiqueta.get("estado", "ERROR")

    if clase in ("CRITICO", "MAYOR"):
        motivos.append(f"Defecto físico {clase.lower()}")
    if estado in ("VENCIDA", "ILEGIBLE", "SIN_FECHA"):
        motivos.append(f"Etiqueta: {estado.lower()}")

    es_nc = len(motivos) > 0
    return es_nc, " · ".join(motivos) if motivos else "Conforme"

# ══════════════════════════════════════════════════════════════════════════════
# GENERADOR DE REPORTE PDF
# ══════════════════════════════════════════════════════════════════════════════
def generar_reporte_pdf(resultados_latas, N, n, c, decision_lote, X, session_id):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    buf = io.BytesIO()
    cv = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    now = datetime.now()
    p_prima = round(X / n, 4) if n > 0 else 0

    # ── Página 1: cabecera y resumen ──────────────────────────────────────
    # Fondo
    cv.setFillColor(colors.HexColor("#0b1120"))
    cv.rect(0, 0, W, H, fill=1, stroke=0)

    # Header negro
    cv.setFillColor(colors.HexColor("#0d1f40"))
    cv.rect(0, H - 90, W, 90, fill=1, stroke=0)
    cv.setFillColor(colors.HexColor("#f0b429"))
    cv.rect(0, H - 92, W, 3, fill=1, stroke=0)
    cv.setFillColor(colors.white)
    cv.setFont("Helvetica-Bold", 16)
    cv.drawString(1.5 * cm, H - 38, "REPORTE DE INSPECCIÓN DE LOTE")
    cv.setFont("Helvetica", 10)
    cv.drawString(1.5 * cm, H - 56, "Distribuidora ANCO S.A.S.  ·  Sistema de Muestreo ISO 2859-1  ·  NCA 2.5%")
    cv.setFont("Helvetica", 9)
    cv.drawString(1.5 * cm, H - 72, f"Generado: {now.strftime('%d/%m/%Y %H:%M:%S')}   ID sesión: {session_id}")

    # Decisión
    color_dec = colors.HexColor("#27c97e") if decision_lote == "ACEPTADO" else colors.HexColor("#e55353")
    cv.setFillColor(color_dec)
    cv.roundRect(W - 6 * cm, H - 80, 4.5 * cm, 46, 4, fill=1, stroke=0)
    cv.setFillColor(colors.black)
    cv.setFont("Helvetica-Bold", 18)
    cv.drawCentredString(W - 3.75 * cm, H - 52, decision_lote)

    # Parámetros del plan
    y = H - 118
    cv.setFillColor(colors.HexColor("#0d1830"))
    cv.rect(1.2 * cm, y - 18, W - 2.4 * cm, 26, fill=1, stroke=0)
    cv.setFillColor(colors.HexColor("#3a7bd5"))
    cv.rect(1.2 * cm, y - 18, 4, 26, fill=1, stroke=0)
    cv.setFont("Helvetica-Bold", 9)
    cv.setFillColor(colors.HexColor("#3a7bd5"))
    params = f"N = {N}   ·   n = {n}   ·   c = {c}   ·   X = {X}   ·   p' = {p_prima:.1%}"
    cv.drawString(1.8 * cm, y - 5, params)

    # Tabla de resultados por lata
    y -= 42
    col_w = [1.2 * cm, 2.0 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm, 4.0 * cm]
    headers = ["#", "LATA", "CUERPO", "CONFIANZA", "ETIQUETA", "DECISIÓN"]
    col_x = [1.2 * cm]
    for w in col_w[:-1]:
        col_x.append(col_x[-1] + w)

    # Header tabla
    cv.setFillColor(colors.HexColor("#1a3a6a"))
    cv.rect(1.2 * cm, y - 6, W - 2.4 * cm, 20, fill=1, stroke=0)
    cv.setFont("Helvetica-Bold", 8)
    cv.setFillColor(colors.HexColor("#7aa3cc"))
    for i, h in enumerate(headers):
        cv.drawString(col_x[i] + 4, y + 5, h)

    y -= 10
    for idx, lata in enumerate(resultados_latas):
        if y < 2 * cm:
            cv.showPage()
            # fondo nueva página
            cv.setFillColor(colors.HexColor("#0b1120"))
            cv.rect(0, 0, W, H, fill=1, stroke=0)
            y = H - 2 * cm

        bg = colors.HexColor("#0d1830") if idx % 2 == 0 else colors.HexColor("#091020")
        cv.setFillColor(bg)
        cv.rect(1.2 * cm, y - 14, W - 2.4 * cm, 18, fill=1, stroke=0)

        nc = lata["no_conforme"]
        clase = lata["cuerpo"].get("clase", "?")
        estado = lata["etiqueta"].get("estado", "?")
        conf_c = lata["cuerpo"].get("confianza", 0)
        dec_txt = "NO CONFORME" if nc else ("OBS" if clase == "MENOR" else "CONFORME")
        dec_color = colors.HexColor("#e55353") if nc else (colors.HexColor("#f0b429") if clase == "MENOR" else colors.HexColor("#27c97e"))

        cv.setFont("Helvetica", 8)
        cv.setFillColor(colors.HexColor("#5a7a9a"))
        cv.drawString(col_x[0] + 4, y - 4, str(idx + 1))
        cv.setFillColor(colors.white)
        cv.drawString(col_x[1] + 4, y - 4, lata["id"])
        cv.setFillColor(colors.HexColor("#c8d6e5"))
        cv.drawString(col_x[2] + 4, y - 4, clase[:12])
        cv.setFillColor(colors.HexColor("#7aa3cc"))
        cv.drawString(col_x[3] + 4, y - 4, f"{conf_c:.0%}")
        cv.setFillColor(colors.HexColor("#c8d6e5"))
        cv.drawString(col_x[4] + 4, y - 4, estado[:12])
        cv.setFillColor(dec_color)
        cv.setFont("Helvetica-Bold", 8)
        cv.drawString(col_x[5] + 4, y - 4, dec_txt)
        y -= 18

    # ── Página 2: CCO ──────────────────────────────────────────────────────
    cv.showPage()
    cv.setFillColor(colors.HexColor("#0b1120"))
    cv.rect(0, 0, W, H, fill=1, stroke=0)

    cv.setFillColor(colors.HexColor("#0d1f40"))
    cv.rect(0, H - 50, W, 50, fill=1, stroke=0)
    cv.setFont("Helvetica-Bold", 13)
    cv.setFillColor(colors.white)
    cv.drawString(1.5 * cm, H - 30, "CURVA CARACTERÍSTICA DE OPERACIÓN (CCO)")
    cv.setFont("Helvetica", 9)
    cv.setFillColor(colors.HexColor("#5a7a9a"))
    cv.drawString(1.5 * cm, H - 44, f"Plan: n={n}, c={c}  ·  NCA=2.5%  ·  Distribución binomial")

    # Dibujar CCO
    margen_izq = 2.5 * cm
    margen_der = W - 1.5 * cm
    margen_inf = 3.5 * cm
    margen_sup = H - 2 * cm
    ancho_g = margen_der - margen_izq
    alto_g = margen_sup - margen_inf - 70

    ps, pas = calcular_cco(n, c)
    alpha, nql = encontrar_riesgos(n, c)

    def px(p): return margen_izq + p * ancho_g
    def py(pa): return margen_inf + pa * alto_g + 70

    # Área bajo la curva (relleno suave)
    path = cv.beginPath()
    path.moveTo(px(0), py(0))
    for p, pa in zip(ps, pas):
        path.lineTo(px(p), py(pa))
    path.lineTo(px(1), py(0))
    path.close()
    cv.setFillColor(colors.HexColor("#0d2040"))
    cv.drawPath(path, fill=1, stroke=0)

    # Línea de la curva
    cv.setStrokeColor(colors.HexColor("#3a7bd5"))
    cv.setLineWidth(2)
    path2 = cv.beginPath()
    path2.moveTo(px(ps[0]), py(pas[0]))
    for p, pa in zip(ps[1:], pas[1:]):
        path2.lineTo(px(p), py(pa))
    cv.drawPath(path2, fill=0, stroke=1)

    # Ejes
    cv.setStrokeColor(colors.HexColor("#1a3a6a"))
    cv.setLineWidth(0.8)
    cv.line(px(0), py(0), px(1), py(0))
    cv.line(px(0), py(0), px(0), py(1))

    # Graduaciones eje X
    cv.setFont("Helvetica", 7)
    cv.setFillColor(colors.HexColor("#5a7a9a"))
    for val in [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        xv = px(val)
        cv.line(xv, py(0) - 3, xv, py(0))
        cv.drawCentredString(xv, py(0) - 12, f"{int(val*100)}%")

    # Graduaciones eje Y
    for val in [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]:
        yv = py(val)
        cv.line(px(0) - 3, yv, px(0), yv)
        cv.drawRightString(px(0) - 6, yv - 3, f"{int(val*100)}%")

    # Etiquetas de ejes
    cv.setFont("Helvetica-Bold", 8)
    cv.setFillColor(colors.HexColor("#7aa3cc"))
    cv.drawCentredString((margen_izq + margen_der) / 2, margen_inf + 42, "Proporción real de defectos p")
    cv.saveState()
    cv.rotate(90)
    cv.drawCentredString(py(0.5), -(margen_izq - 18), "Pa(p) – Prob. de aceptación")
    cv.restoreState()

    # Línea NCA vertical
    cv.setStrokeColor(colors.HexColor("#f0b429"))
    cv.setLineWidth(1)
    cv.setDash([4, 3])
    cv.line(px(0.025), py(0), px(0.025), py(pa_binomial(0.025, n, c)))
    cv.setDash()
    cv.setFillColor(colors.HexColor("#f0b429"))
    cv.setFont("Helvetica-Bold", 7)
    cv.drawCentredString(px(0.025), py(0) - 22, f"NCA\n2.5%")
    pa_nca = pa_binomial(0.025, n, c)
    cv.drawCentredString(px(0.025), py(pa_nca) + 8, f"1−α={pa_nca:.0%}")

    # Línea NQL vertical
    if nql:
        pa_nql = pa_binomial(nql, n, c)
        cv.setStrokeColor(colors.HexColor("#e55353"))
        cv.setLineWidth(1)
        cv.setDash([4, 3])
        cv.line(px(nql), py(0), px(nql), py(pa_nql))
        cv.setDash()
        cv.setFillColor(colors.HexColor("#e55353"))
        cv.drawCentredString(px(nql), py(0) - 22, f"NQL\n{nql:.0%}")
        cv.drawCentredString(px(nql), py(pa_nql) + 8, f"β={pa_nql:.0%}")

    # Leyenda riesgos
    ly = margen_inf + 20
    cv.setFont("Helvetica", 8)
    cv.setFillColor(colors.HexColor("#f0b429"))
    cv.drawString(margen_izq, ly, f"α (riesgo productor) = {alpha:.1%}  |  rechazar lote bueno (p=NCA)")
    if nql:
        cv.setFillColor(colors.HexColor("#e55353"))
        cv.drawString(margen_izq, ly - 14, f"β (riesgo consumidor) ≤ 10%  |  aceptar lote malo (p=NQL={nql:.0%})")

    # ── Página 3: detalle de no conformes ─────────────────────────────────
    no_conformes = [l for l in resultados_latas if l["no_conforme"]]
    if no_conformes:
        cv.showPage()
        cv.setFillColor(colors.HexColor("#0b1120"))
        cv.rect(0, 0, W, H, fill=1, stroke=0)
        cv.setFillColor(colors.HexColor("#0d1f40"))
        cv.rect(0, H - 50, W, 50, fill=1, stroke=0)
        cv.setFont("Helvetica-Bold", 13)
        cv.setFillColor(colors.HexColor("#e55353"))
        cv.drawString(1.5 * cm, H - 30, f"EVIDENCIA – LATAS NO CONFORMES ({len(no_conformes)} de {n})")
        cv.setFont("Helvetica", 9)
        cv.setFillColor(colors.HexColor("#5a7a9a"))
        cv.drawString(1.5 * cm, H - 44, "Motivos de rechazo por unidad")

        y = H - 72
        for lata in no_conformes:
            if y < 3 * cm:
                cv.showPage()
                cv.setFillColor(colors.HexColor("#0b1120"))
                cv.rect(0, 0, W, H, fill=1, stroke=0)
                y = H - 2 * cm

            cv.setFillColor(colors.HexColor("#1a0707"))
            cv.roundRect(1.2 * cm, y - 38, W - 2.4 * cm, 40, 2, fill=1, stroke=0)
            cv.setStrokeColor(colors.HexColor("#5a1a1a"))
            cv.setLineWidth(0.5)
            cv.roundRect(1.2 * cm, y - 38, W - 2.4 * cm, 40, 2, fill=0, stroke=1)

            cv.setFont("Helvetica-Bold", 10)
            cv.setFillColor(colors.HexColor("#e55353"))
            cv.drawString(1.8 * cm, y - 4, f"Lata {lata['id']}")
            cv.setFont("Helvetica", 8)
            cv.setFillColor(colors.HexColor("#c8d6e5"))
            cv.drawString(1.8 * cm, y - 18, f"Cuerpo: {lata['cuerpo'].get('clase','?')} ({lata['cuerpo'].get('confianza',0):.0%}) – {lata['cuerpo'].get('descripcion','')[:60]}")
            cv.drawString(1.8 * cm, y - 30, f"Etiqueta: {lata['etiqueta'].get('estado','?')} – {lata['etiqueta'].get('descripcion','')[:60]}")
            y -= 50

    cv.save()
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ══════════════════════════════════════════════════════════════════════════════
if "historial_lotes" not in st.session_state:
    st.session_state.historial_lotes = []
if "resultados_actuales" not in st.session_state:
    st.session_state.resultados_actuales = []
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")

# ══════════════════════════════════════════════════════════════════════════════
# API KEY
# ══════════════════════════════════════════════════════════════════════════════
def get_api_key():
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if k:
        return k
    try:
        return st.secrets["ANTHROPIC_API_KEY"].strip()
    except Exception:
        return ""

def make_client(key):
    return anthropic.Anthropic(api_key=key) if key else None

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙ Configuración")
    api_key_in = st.text_input("API Key Anthropic", type="password", placeholder="sk-ant-api03-...")
    if api_key_in:
        os.environ["ANTHROPIC_API_KEY"] = api_key_in.strip()

    st.markdown("---")
    modelo_sel = st.selectbox(
        "Modelo Claude",
        options=list(MODELOS),
        format_func=lambda m: "Sonnet 4.6 (recomendado)" if "sonnet" in m else "Haiku 4.5 (rápido)",
    )
    intentos = st.select_slider(
        "Votaciones por lata",
        options=[1, 2, 3],
        value=3,
        help="3 votaciones = máxima precisión (triple consenso). 1 = más rápido.",
    )
    mostrar_fotos = st.checkbox("Mostrar fotos en resultados", value=True)

    st.markdown("---")
    st.markdown("**Historial de lotes**")
    if st.session_state.historial_lotes:
        for lote in reversed(st.session_state.historial_lotes[-8:]):
            color_d = "#27c97e" if lote["decision"] == "ACEPTADO" else "#e55353"
            st.markdown(
                f'<div class="hist-item">'
                f'<span class="hist-id">Lote {lote["id"]}</span>'
                f'<span class="hist-stat" style="color:{color_d}">{lote["decision"]} X={lote["X"]}/{lote["n"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.button("🗑 Limpiar historial"):
            st.session_state.historial_lotes = []
            st.session_state.resultados_actuales = []
            st.rerun()
    else:
        st.caption("Sin lotes inspeccionados")

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">🥫 SISTEMA DE INSPECCIÓN DE LOTES – ANCO S.A.S.</div>'
    '<p class="hero-sub">ISO 2859-1 · NCA 2.5% · YOLOv8n (cuerpo) + Claude Vision (etiqueta) · Gerencia y Control de Calidad</p>'
    '</div>',
    unsafe_allow_html=True,
)

api_key = get_api_key()
client = make_client(api_key)

if not api_key:
    st.warning("Ingresa tu API Key de Anthropic en el panel izquierdo para comenzar.", icon="🔑")
else:
    st.success("API Key detectada.", icon="✅")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_plan, tab_insp, tab_hist = st.tabs([
    "📐 PLAN DE MUESTREO",
    "🔬 INSPECCIÓN",
    "📊 HISTORIAL & CCO",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – PLAN DE MUESTREO
# ─────────────────────────────────────────────────────────────────────────────
with tab_plan:
    st.markdown("#### Determinación del plan según ISO 2859-1")
    col1, col2 = st.columns([1, 2])

    with col1:
        N_input = st.number_input(
            "Tamaño del lote N (unidades)",
            min_value=2, max_value=500000, value=60, step=1,
        )

    letra, n_auto, c_auto = iso_plan(N_input)
    alpha_plan, nql_plan = encontrar_riesgos(n_auto, c_auto)
    pa_nca = pa_binomial(0.025, n_auto, c_auto)

    with col2:
        st.markdown(
            f'<div class="plan-box">'
            f'<div class="plan-row">'
            f'<span><span class="plan-param">LETRA CÓDIGO: </span><span class="plan-val">{letra}</span></span>'
            f'<span><span class="plan-param">MUESTRA n = </span><span class="plan-val">{n_auto}</span></span>'
            f'<span><span class="plan-param">ACEPTAR si X ≤ </span><span class="plan-val">{c_auto}</span></span>'
            f'<span><span class="plan-param">RECHAZAR si X > </span><span class="plan-val">{c_auto}</span></span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="metrics-row">'
        f'<div class="met neu"><span class="met-val met-neu">{N_input}</span><div class="met-lbl">Tamaño lote N</div></div>'
        f'<div class="met neu"><span class="met-val met-neu">{n_auto}</span><div class="met-lbl">Muestra n</div></div>'
        f'<div class="met warn"><span class="met-val met-warn">{c_auto}</span><div class="met-lbl">Núm. aceptación c</div></div>'
        f'<div class="met ok"><span class="met-val met-ok">{pa_nca:.0%}</span><div class="met-lbl">Pa en NCA (1−α)</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.info(
        f"**Interpretación:** Para un lote de **{N_input}** latas, debes inspeccionar **{n_auto}** latas. "
        f"Si encuentras **{c_auto} o menos** defectuosas → ACEPTAS el lote. "
        f"Si encuentras **{c_auto+1} o más** → RECHAZAS. "
        f"La probabilidad de aceptar un lote con exactamente 2.5% de defectos es **{pa_nca:.1%}** (α = {alpha_plan:.1%}).",
        icon="📐",
    )

    st.markdown("---")
    st.markdown("#### Reglas de clasificación por lata")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**Defectos de cuerpo (YOLOv8n)**")
        st.markdown(
            "| Clase | Cuenta como NC |\n"
            "|---|---|\n"
            "| CRÍTICO | ✅ Sí |\n"
            "| MAYOR | ✅ Sí |\n"
            "| MENOR | ⚠️ Observación |\n"
            "| CONFORME | ❌ No |"
        )
    with col_r2:
        st.markdown("**Defectos de etiqueta**")
        st.markdown(
            "| Estado | Cuenta como NC |\n"
            "|---|---|\n"
            "| VENCIDA | ✅ Sí |\n"
            "| ILEGIBLE | ✅ Sí |\n"
            "| SIN_FECHA | ✅ Sí |\n"
            "| VIGENTE | ❌ No |"
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – INSPECCIÓN
# ─────────────────────────────────────────────────────────────────────────────
with tab_insp:
    if not client:
        st.error("Configura la API Key primero.", icon="🔑")
        st.stop()

    st.markdown("#### Configuración del lote a inspeccionar")
    c1, c2, c3 = st.columns(3)
    with c1:
        N_lote = st.number_input("N – Tamaño del lote", min_value=2, max_value=500000, value=60, step=1, key="N_insp")
    with c2:
        _, n_lote, c_lote = iso_plan(N_lote)
        st.metric("n – Latas a inspeccionar", n_lote)
    with c3:
        st.metric("c – Número de aceptación", c_lote)

    st.markdown("---")
    st.markdown(
        f"#### Subir imágenes de la muestra  "
        f"<span style='color:#5a7a9a;font-size:13px;'>({n_lote} latas · 2 fotos por lata)</span>",
        unsafe_allow_html=True,
    )

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        fotos_cuerpo = st.file_uploader(
            f"📸 Fotos del CUERPO (cuerpo_001, cuerpo_002... — {n_lote} fotos)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key="cuerpo",
        )
    with col_up2:
        fotos_etiqueta = st.file_uploader(
            f"🏷️ Fotos de la ETIQUETA (etiqueta_001, etiqueta_002... — {n_lote} fotos)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key="etiqueta",
        )

    # Info de emparejamiento
    if fotos_cuerpo or fotos_etiqueta:
        n_c = len(fotos_cuerpo) if fotos_cuerpo else 0
        n_e = len(fotos_etiqueta) if fotos_etiqueta else 0
        pares = min(n_c, n_e)
        if n_c != n_e:
            st.warning(f"Cuerpos: {n_c} · Etiquetas: {n_e} → Se analizarán {pares} pares.", icon="⚠️")
        else:
            st.success(f"{pares} pares emparejados listos para analizar.", icon="✅")

    # Botón de análisis
    puede_analizar = (
        client and
        fotos_cuerpo and fotos_etiqueta and
        len(fotos_cuerpo) > 0 and len(fotos_etiqueta) > 0
    )

    if st.button("🔬 INICIAR INSPECCIÓN", disabled=not puede_analizar, type="primary"):

        # Ordenar por nombre para emparejar
        fotos_cuerpo_s = sorted(fotos_cuerpo, key=lambda f: f.name)
        fotos_etiqueta_s = sorted(fotos_etiqueta, key=lambda f: f.name)
        pares_total = min(len(fotos_cuerpo_s), len(fotos_etiqueta_s))

        resultados = []
        X = 0
        obs_count = 0

        # Progress
        progress_bar = st.progress(0, text="Iniciando inspección...")
        status_text = st.empty()
        col_live1, col_live2 = st.columns(2)

        for i in range(pares_total):
            lata_id = f"{str(i+1).zfill(3)}"
            pct = (i) / pares_total
            progress_bar.progress(pct, text=f"Analizando lata {lata_id} de {pares_total}...")

            img_cuerpo = Image.open(fotos_cuerpo_s[i]).convert("RGB")
            img_etiqueta = Image.open(fotos_etiqueta_s[i]).convert("RGB")

            # ── CUERPO: YOLO si está disponible, Claude Vision como fallback ──
            status_text.markdown(f"`[{lata_id}]` Analizando cuerpo con YOLO...")
            yolo_model = cargar_yolo()
            if yolo_model is not None:
                res_cuerpo = analizar_cuerpo_yolo(yolo_model, img_cuerpo)
            else:
                # Fallback a Claude Vision si YOLO no está disponible
                status_text.markdown(f"`[{lata_id}]` YOLO no disponible → usando Claude Vision para cuerpo...")
                res_cuerpo = analizar_con_consenso(client, img_cuerpo, PROMPT_CLAUDE_CUERPO_FALLBACK, modelo_sel, intentos)

            # ── ETIQUETA: siempre Claude Vision (1 sola llamada, sin triple votación si YOLO activo) ──
            status_text.markdown(f"`[{lata_id}]` Analizando etiqueta con Claude Vision...")
            votos_etiqueta = 1 if yolo_model is not None else intentos
            res_etiqueta = analizar_con_consenso(client, img_etiqueta, PROMPT_ETIQUETA, modelo_sel, votos_etiqueta)

            nc, motivo = es_no_conforme(res_cuerpo, res_etiqueta)
            if nc:
                X += 1
            elif res_cuerpo.get("clase") == "MENOR":
                obs_count += 1

            resultados.append({
                "id": lata_id,
                "cuerpo": res_cuerpo,
                "etiqueta": res_etiqueta,
                "no_conforme": nc,
                "motivo": motivo,
                "img_cuerpo": img_cuerpo,
                "img_etiqueta": img_etiqueta,
            })

        progress_bar.progress(1.0, text="Inspección completada.")
        status_text.empty()

        # Decisión del lote
        decision = "ACEPTADO" if X <= c_lote else "RECHAZADO"
        p_prima = X / pares_total if pares_total > 0 else 0

        st.session_state.resultados_actuales = resultados
        st.session_state.historial_lotes.append({
            "id": datetime.now().strftime("%H%M%S"),
            "N": N_lote,
            "n": pares_total,
            "c": c_lote,
            "X": X,
            "decision": decision,
        })

        # ── Decisión del lote ──
        clase_dec = "decision-acepta" if decision == "ACEPTADO" else "decision-rechaza"
        color_val = "#27c97e" if decision == "ACEPTADO" else "#e55353"
        st.markdown(
            f'<div class="decision-box {clase_dec}">'
            f'<div class="decision-title" style="color:{color_val}">{decision}</div>'
            f'<div class="decision-sub">X = {X}  {"≤" if decision == "ACEPTADO" else ">"} c = {c_lote}  '
            f'·  p\' = {p_prima:.1%}  ·  {pares_total} latas inspeccionadas</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Métricas resumen
        st.markdown(
            f'<div class="metrics-row">'
            f'<div class="met {"ok" if decision=="ACEPTADO" else "bad"}">'
            f'<span class="met-val {"met-ok" if decision=="ACEPTADO" else "met-bad"}">{X}</span>'
            f'<div class="met-lbl">No conformes X</div></div>'
            f'<div class="met neu"><span class="met-val met-neu">{pares_total - X - obs_count}</span>'
            f'<div class="met-lbl">Conformes</div></div>'
            f'<div class="met warn"><span class="met-val met-warn">{obs_count}</span>'
            f'<div class="met-lbl">Observaciones</div></div>'
            f'<div class="met neu"><span class="met-val met-neu">{p_prima:.1%}</span>'
            f'<div class="met-lbl">p\' estimada</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Detalle por lata ──
        st.markdown("---")
        st.markdown("#### Detalle por lata")

        for lata in resultados:
            nc = lata["no_conforme"]
            clase = lata["cuerpo"].get("clase", "?")
            estado = lata["etiqueta"].get("estado", "?")
            conf_c = lata["cuerpo"].get("confianza", 0)
            conf_e = lata["etiqueta"].get("confianza", 0)

            if nc:
                badge = f'<span class="badge badge-nc">NO CONFORME</span>'
            elif clase == "MENOR":
                badge = f'<span class="badge badge-obs">OBSERVACIÓN</span>'
            else:
                badge = f'<span class="badge badge-ok">CONFORME</span>'

            fuente_cuerpo = lata['cuerpo'].get('fuente', 'Claude')
            detalle = (
                f"[{fuente_cuerpo}] Cuerpo: <b>{clase}</b> ({conf_c:.0%}) – {lata['cuerpo'].get('descripcion','')[:60]} | "
                f"[Claude] Etiqueta: <b>{estado}</b> ({conf_e:.0%})"
                + (f" · {lata['etiqueta'].get('fecha_leida','')}" if lata["etiqueta"].get("fecha_leida") else "")
            )

            st.markdown(
                f'<div class="lata-card">'
                f'<div class="lata-id">#{lata["id"]}</div>'
                f'<div class="lata-detail">{detalle}</div>'
                f'{badge}'
                f'</div>',
                unsafe_allow_html=True,
            )

            if mostrar_fotos and nc:
                c_img1, c_img2 = st.columns(2)
                with c_img1:
                    boxes = lata['cuerpo'].get('boxes', [])
                    if boxes:
                        img_ann = dibujar_boxes(lata["img_cuerpo"], boxes)
                        st.image(img_ann, caption=f"Cuerpo {lata['id']} (YOLO)", width=220)
                    else:
                        st.image(lata["img_cuerpo"], caption=f"Cuerpo {lata['id']}", width=220)
                with c_img2:
                    st.image(lata["img_etiqueta"], caption=f"Etiqueta {lata['id']} (Claude)", width=220)

        # ── Reporte PDF ──
        st.markdown("---")
        pdf_buf = generar_reporte_pdf(resultados, N_lote, pares_total, c_lote, decision, X, st.session_state.session_id)
        st.download_button(
            label="📄 Descargar Reporte PDF completo",
            data=pdf_buf,
            file_name=f"reporte_ANCO_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )

    elif st.session_state.resultados_actuales:
        st.info("Resultados del último análisis disponibles en la pestaña Historial & CCO.", icon="📊")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – HISTORIAL & CCO
# ─────────────────────────────────────────────────────────────────────────────
with tab_hist:
    st.markdown("#### Curva Característica de Operación (CCO)")

    if st.session_state.historial_lotes:
        ultimo = st.session_state.historial_lotes[-1]
        n_cco = ultimo["n"]
        c_cco = ultimo["c"]
        n_lote_cco = ultimo["N"]
    else:
        # CCO con el plan calculado de N=900 por defecto
        _, n_cco, c_cco = iso_plan(900)
        n_lote_cco = 900

    ps, pas = calcular_cco(n_cco, c_cco)
    alpha, nql = encontrar_riesgos(n_cco, c_cco)

    # Dibujar CCO con matplotlib dentro de streamlit
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor("#0b1120")
        ax.set_facecolor("#0d1830")

        ax.fill_between(ps, pas, alpha=0.15, color="#3a7bd5")
        ax.plot(ps, pas, color="#3a7bd5", linewidth=2.5, label=f"Pa(p)  n={n_cco}, c={c_cco}")
        ax.axhline(y=0.10, color="#e55353", linestyle="--", linewidth=1, alpha=0.6, label="β = 10%")
        ax.axvline(x=0.025, color="#f0b429", linestyle="--", linewidth=1, alpha=0.8, label="NCA = 2.5%")

        if nql:
            ax.axvline(x=nql, color="#e55353", linestyle=":", linewidth=1, alpha=0.8, label=f"NQL ≈ {nql:.0%}")
            ax.annotate(
                f"β≤10%\nNQL={nql:.0%}",
                xy=(nql, 0.10), xytext=(nql + 0.02, 0.25),
                arrowprops=dict(arrowstyle="->", color="#e55353"),
                color="#e55353", fontsize=8,
            )

        pa_nca = pa_binomial(0.025, n_cco, c_cco)
        ax.annotate(
            f"1−α={pa_nca:.0%}\nNCA=2.5%",
            xy=(0.025, pa_nca), xytext=(0.07, pa_nca - 0.12),
            arrowprops=dict(arrowstyle="->", color="#f0b429"),
            color="#f0b429", fontsize=8,
        )

        # Marcar p' del último lote si existe
        if st.session_state.historial_lotes:
            ultimo = st.session_state.historial_lotes[-1]
            p_prima_lote = ultimo["X"] / ultimo["n"] if ultimo["n"] > 0 else 0
            pa_lote = pa_binomial(p_prima_lote, n_cco, c_cco)
            color_punto = "#27c97e" if ultimo["decision"] == "ACEPTADO" else "#e55353"
            ax.scatter([p_prima_lote], [pa_lote], color=color_punto, zorder=5, s=80,
                       label=f"Último lote p'={p_prima_lote:.1%} → {ultimo['decision']}")

        ax.set_xlabel("Proporción real de defectos p", color="#7aa3cc", fontsize=10)
        ax.set_ylabel("Pa(p) – Probabilidad de aceptación", color="#7aa3cc", fontsize=10)
        ax.set_title(f"CCO  ·  Plan n={n_cco}, c={c_cco}  ·  ISO 2859-1  ·  NCA=2.5%",
                     color="#c8d6e5", fontsize=11, pad=10)
        ax.tick_params(colors="#5a7a9a")
        ax.spines[:].set_color("#1a3a6a")
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(True, color="#1a2744", linewidth=0.5, alpha=0.7)
        ax.legend(loc="upper right", facecolor="#0d1830", edgecolor="#1a3a6a",
                  labelcolor="#c8d6e5", fontsize=8)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except ImportError:
        st.warning("Instala matplotlib para ver la CCO: `pip install matplotlib`", icon="⚠️")

    # Tabla de riesgos
    nql_str = f"{nql:.0%}" if nql else "N/D"
    st.markdown(
        f'<div class="metrics-row">'
        f'<div class="met warn"><span class="met-val met-warn">{alpha:.1%}</span><div class="met-lbl">α – Riesgo productor</div></div>'
        f'<div class="met bad"><span class="met-val met-bad">≤10%</span><div class="met-lbl">β – Riesgo consumidor</div></div>'
        f'<div class="met ok"><span class="met-val met-ok">{pa_nca:.0%}</span><div class="met-lbl">Pa en NCA</div></div>'
        f'<div class="met neu"><span class="met-val met-neu">{nql_str}</span><div class="met-lbl">NQL estimado</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Historial de lotes
    if st.session_state.historial_lotes:
        st.markdown("---")
        st.markdown("#### Historial de lotes")
        filas = []
        for lote in reversed(st.session_state.historial_lotes):
            filas.append({
                "Lote ID": lote["id"],
                "N": lote["N"],
                "n": lote["n"],
                "c": lote["c"],
                "X": lote["X"],
                "p'": f"{lote['X']/lote['n']:.1%}" if lote["n"] > 0 else "—",
                "Decisión": lote["decision"],
            })
        st.dataframe(filas, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="footer">ANCO S.A.S. · ISO 2859-1 · NCA 2.5% · YOLOv8n + Claude Vision · Gerencia y Control de Calidad · UNICAUCA 2026</div>',
    unsafe_allow_html=True,
)
