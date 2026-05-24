"""
Sistema de Inspección de Lotes – Distribuidora ANCO S.A.S.
ISO 2859-1 · YOLOv8n + Claude Vision (retroalimentación opcional)
"""

import streamlit as st
import anthropic
import base64, io, os, json, re, time, math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PIL import Image, ImageEnhance, ImageDraw
from collections import Counter
from datetime import datetime

# ── YOLO / cv2 / numpy ────────────────────────────────────────────────────────
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # forzar CPU antes de importar torch
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("YOLO_VERBOSE", "False")

YOLO_DISPONIBLE = False
YOLO_ERROR_MSG = ""
np = cv2 = ort = None

try:
    import numpy as np
    YOLO_DISPONIBLE = True
except Exception as _e:
    YOLO_ERROR_MSG = f"numpy: {_e}"

if YOLO_DISPONIBLE:
    try:
        import cv2
    except Exception as _e:
        YOLO_DISPONIBLE = False
        YOLO_ERROR_MSG = f"cv2: {_e}"

if YOLO_DISPONIBLE:
    try:
        import onnxruntime as ort
    except Exception as _e:
        YOLO_DISPONIBLE = False
        YOLO_ERROR_MSG = f"onnxruntime: {_e}"
        ort = None

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ANCO – Inspección de Lotes", page_icon="🥫", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
*,*::before,*::after{box-sizing:border-box}
.stApp{background:#0b1120;color:#c8d6e5;font-family:'IBM Plex Sans',sans-serif}
section[data-testid='stSidebar']{background:#080e1a;border-right:1px solid #1a2744}
.stTabs [data-baseweb="tab-list"]{background:#0d1830;border-bottom:2px solid #1a2744;gap:4px}
.stTabs [data-baseweb="tab"]{color:#5a7a9a!important;font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;padding:10px 20px;border-radius:0!important}
.stTabs [aria-selected="true"]{color:#f0b429!important;border-bottom:2px solid #f0b429!important;background:transparent!important}
.hero{background:linear-gradient(135deg,#0d1f40 0%,#0b1120 100%);border:1px solid #1a3a6a;border-left:4px solid #f0b429;border-radius:4px;padding:1.5rem 2rem;margin-bottom:1.5rem}
.hero-title{font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:600;color:#f0b429;margin:0 0 4px 0;letter-spacing:1px}
.hero-sub{font-size:.85rem;color:#5a7a9a;margin:0;letter-spacing:.5px}
.metrics-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}
.met{background:#0d1830;border:1px solid #1a2744;border-top:3px solid;border-radius:4px;padding:14px;text-align:center}
.met.ok{border-top-color:#27c97e}.met.warn{border-top-color:#f0b429}.met.bad{border-top-color:#e55353}.met.neu{border-top-color:#3a7bd5}
.met-val{font-family:'IBM Plex Mono',monospace;font-size:1.6rem;font-weight:600;display:block}
.met-ok{color:#27c97e}.met-warn{color:#f0b429}.met-bad{color:#e55353}.met-neu{color:#3a7bd5}
.met-lbl{font-size:10px;color:#4a6a8a;letter-spacing:1px;text-transform:uppercase;margin-top:4px}
.lata-card{background:#0d1830;border:1px solid #1a2744;border-radius:4px;padding:14px;margin-bottom:10px;display:flex;align-items:flex-start;gap:14px}
.lata-id{font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:600;color:#7aa3cc;min-width:60px}
.lata-detail{flex:1;font-size:13px;color:#7a9ab8}
.badge{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;padding:4px 12px;border-radius:2px;letter-spacing:1px;white-space:nowrap}
.badge-ok{background:#0a2a1a;color:#27c97e;border:1px solid #1a5a3a}
.badge-nc{background:#2a0a0a;color:#e55353;border:1px solid #5a1a1a}
.badge-obs{background:#2a200a;color:#f0b429;border:1px solid #5a4010}
.corr-tag{font-size:10px;font-family:'IBM Plex Mono',monospace;background:#1a0a2a;color:#a78bfa;border:1px solid #4a2a7a;border-radius:2px;padding:2px 6px;margin-left:6px}
.decision-box{border-radius:6px;padding:2rem;text-align:center;margin:1rem 0}
.decision-acepta{background:#071a0f;border:2px solid #27c97e}
.decision-rechaza{background:#1a0707;border:2px solid #e55353}
.decision-title{font-family:'IBM Plex Mono',monospace;font-size:2.2rem;font-weight:600;letter-spacing:3px}
.decision-sub{font-size:13px;color:#7a9ab8;margin-top:8px}
.hist-item{background:#0d1830;border:1px solid #1a2744;border-radius:3px;padding:8px 10px;margin-bottom:6px;font-family:'IBM Plex Mono',monospace;font-size:12px;display:flex;justify-content:space-between;align-items:center}
.hist-id{color:#7aa3cc}
.plan-box{background:#0d1830;border:1px solid #1a2744;border-left:3px solid #3a7bd5;border-radius:4px;padding:12px 16px;font-family:'IBM Plex Mono',monospace;font-size:13px;margin:12px 0}
.plan-row{display:flex;gap:24px;flex-wrap:wrap}
.plan-param{color:#3a7bd5}.plan-val{color:#f0b429;font-weight:600}
.retro-box{background:#0d0a1a;border:1px solid #3a2a5a;border-left:3px solid #a78bfa;border-radius:4px;padding:10px 14px;margin:6px 0;font-size:12px;font-family:'IBM Plex Mono',monospace}
.stButton>button{background:#0d1830;border:1px solid #1a3a6a;color:#c8d6e5;font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:1px;border-radius:3px;transition:all .15s}
.stButton>button:hover{border-color:#f0b429;color:#f0b429;background:#1a2a10}
.footer{text-align:center;color:#1a3a5a;font-size:11px;font-family:'IBM Plex Mono',monospace;margin-top:3rem;padding-top:1rem;border-top:1px solid #1a2744;letter-spacing:1px}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ISO 2859-1
# ══════════════════════════════════════════════════════════════════════════════
ISO_LETRA = [
    (2,8,"A"),(9,15,"B"),(16,25,"C"),(26,50,"D"),(51,90,"E"),
    (91,150,"F"),(151,280,"G"),(281,500,"H"),(501,1200,"J"),
    (1201,3200,"K"),(3201,10000,"L"),
]
ISO_PLAN = {
    "A":(2,0),"B":(3,0),"C":(5,0),"D":(8,0),"E":(13,1),
    "F":(20,1),"G":(32,2),"H":(50,3),"J":(80,5),"K":(125,7),"L":(200,10),
}

def iso_letra(N):
    for lo,hi,l in ISO_LETRA:
        if lo<=N<=hi: return l
    return "L"

def iso_plan(N):
    l=iso_letra(N); n,c=ISO_PLAN[l]; return l,min(n,N),c

# ══════════════════════════════════════════════════════════════════════════════
# CCO
# ══════════════════════════════════════════════════════════════════════════════
def comb(n,k):
    if k<0 or k>n: return 0
    if k==0 or k==n: return 1
    k=min(k,n-k); r=1
    for i in range(k): r=r*(n-i)//(i+1)
    return r

def pa_bin(p,n,c):
    if p<=0: return 1.0
    if p>=1: return 0.0
    return min(max(sum(comb(n,x)*(p**x)*((1-p)**(n-x)) for x in range(c+1)),0.0),1.0)

def calcular_cco(n,c,pts=60):
    ps=[i/pts for i in range(pts+1)]
    return ps,[pa_bin(p,n,c) for p in ps]

def riesgos(n,c,nca=0.025,bo=0.10):
    alpha=1-pa_bin(nca,n,c); nql=None
    for i in range(1,101):
        if pa_bin(i/100,n,c)<=bo: nql=i/100; break
    return alpha,nql

# ══════════════════════════════════════════════════════════════════════════════
# YOLO
# ══════════════════════════════════════════════════════════════════════════════
# Busca el modelo en múltiples ubicaciones posibles
import os as _os2, sys as _sys

def _encontrar_modelo(nombre):
    # Buscar en directorio actual, directorio del script, y rutas de Streamlit Cloud
    candidatos = [
        nombre,
        _os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), nombre),
        _os2.path.join("/mount/src/appiagerencia", nombre),
        _os2.path.join(_os2.getcwd(), nombre),
    ]
    for c in candidatos:
        if _os2.path.exists(c):
            return c
    return None

_onnx = _encontrar_modelo("best_latas_defectos.onnx")
_pt   = _encontrar_modelo("best_latas_defectos.pt")
YOLO_PATH    = _onnx if _onnx else (_pt if _pt else "best_latas_defectos.onnx")
YOLO_ES_ONNX = YOLO_PATH.endswith(".onnx")
YOLO_CONF_THR = 0.50
YOLO_IMGSZ    = 416
YOLO_CLASES   = {"Critical Defect":"CRITICO","Major Defect":"MAYOR","Minor Defect":"MENOR","No defect":"CONFORME"}
YOLO_NOMBRES  = {0:"Critical Defect", 1:"Major Defect", 2:"Minor Defect", 3:"No defect"}

@st.cache_resource
def cargar_yolo():
    if not YOLO_DISPONIBLE: return None
    if not os.path.exists(YOLO_PATH) and not os.path.isfile(YOLO_PATH): return None
    if YOLO_ES_ONNX:
        if ort is None: return None
        try:
            return ort.InferenceSession(YOLO_PATH, providers=["CPUExecutionProvider"])
        except Exception:
            return None
    else:
        # Fallback: intentar cargar .pt con ultralytics
        try:
            from ultralytics import YOLO as _YOLO
            return _YOLO(YOLO_PATH)
        except Exception:
            return None

def _nms(boxes, scores, iou_thr=0.45):
    if len(boxes)==0: return []
    x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while order.size>0:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order=order[1:][iou<=iou_thr]
    return keep

def analizar_yolo(modelo, img: Image.Image) -> dict:
    if not YOLO_DISPONIBLE or np is None:
        return {"clase":"ERROR","confianza":0.0,"descripcion":"ONNX no disponible","boxes":[],"fuente":"YOLO"}
    # Si es modelo ultralytics (.pt) usar su API directamente
    if not YOLO_ES_ONNX:
        try:
            img_np = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            results = modelo(img_np, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_THR, verbose=False)
            dets=[]
            for r in results:
                for box in r.boxes:
                    cid=int(box.cls[0]); cn=modelo.names[cid]; cf=float(box.conf[0])
                    x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
                    dets.append({"clase_raw":cn,"clase":YOLO_CLASES.get(cn,"CONFORME"),"confianza":round(cf,3),"box":(x1,y1,x2,y2)})
            if not dets:
                return {"clase":"CONFORME","confianza":0.92,"descripcion":"Sin defectos detectados","boxes":[],"fuente":"YOLO(.pt)"}
            prio={"CRITICO":4,"MAYOR":3,"MENOR":2,"CONFORME":1,"ERROR":0}
            dets.sort(key=lambda d:(prio.get(d["clase"],0),d["confianza"]),reverse=True)
            m=dets[0]
            return {"clase":m["clase"],"confianza":m["confianza"],"descripcion":f"{m['clase_raw']} {m['confianza']:.0%}","boxes":[(d["box"],d["clase"],d["confianza"]) for d in dets],"fuente":"YOLO(.pt)"}
        except Exception as e:
            return {"clase":"ERROR","confianza":0.0,"descripcion":f"Error .pt: {e}","boxes":[],"fuente":"YOLO(.pt)"}
    # ONNX path
    sess = modelo
    W,H=img.size
    # Preprocesar: resize 416x416, normalizar, NCHW
    img_r=img.convert("RGB").resize((YOLO_IMGSZ,YOLO_IMGSZ),Image.LANCZOS)
    inp=np.array(img_r,dtype=np.float32)/255.0
    inp=inp.transpose(2,0,1)[np.newaxis]
    try:
        iname=sess.get_inputs()[0].name
        raw=sess.run(None,{iname:inp})[0]  # [1, 8, anchors]
    except Exception as e:
        return {"clase":"ERROR","confianza":0.0,"descripcion":f"Error ONNX: {e}","boxes":[],"fuente":"YOLO"}
    pred=raw[0].T  # (anchors, 8) = 4coords + 4clases
    boxes_l,scores_l,cids_l=[],[],[]
    for row in pred:
        cs=row[4:]; cid=int(np.argmax(cs)); cf=float(cs[cid])
        if cf<YOLO_CONF_THR: continue
        cx,cy,w,h=row[:4]
        x1=int((cx-w/2)/YOLO_IMGSZ*W); y1=int((cy-h/2)/YOLO_IMGSZ*H)
        x2=int((cx+w/2)/YOLO_IMGSZ*W); y2=int((cy+h/2)/YOLO_IMGSZ*H)
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(W,x2),min(H,y2)
        boxes_l.append([x1,y1,x2,y2]); scores_l.append(cf); cids_l.append(cid)
    if not boxes_l:
        return {"clase":"CONFORME","confianza":0.92,"descripcion":"Sin defectos detectados","boxes":[],"fuente":"YOLO"}
    ba=np.array(boxes_l); sa=np.array(scores_l)
    keep=_nms(ba,sa)
    dets=[]
    for i in keep:
        cn=YOLO_NOMBRES.get(cids_l[i],"No defect")
        dets.append({"clase_raw":cn,"clase":YOLO_CLASES.get(cn,"CONFORME"),"confianza":round(float(sa[i]),3),"box":tuple(ba[i].tolist())})
    prio={"CRITICO":4,"MAYOR":3,"MENOR":2,"CONFORME":1,"ERROR":0}
    dets.sort(key=lambda d:(prio.get(d["clase"],0),d["confianza"]),reverse=True)
    m=dets[0]
    desc={"CRITICO":f"Defecto crítico ({m['clase_raw']}) {m['confianza']:.0%}","MAYOR":f"Defecto mayor ({m['clase_raw']}) {m['confianza']:.0%}","MENOR":f"Defecto menor ({m['clase_raw']}) {m['confianza']:.0%}","CONFORME":f"Conforme {m['confianza']:.0%}"}
    return {"clase":m["clase"],"confianza":m["confianza"],"descripcion":desc.get(m["clase"],""),"boxes":[(tuple(d["box"]),d["clase"],d["confianza"]) for d in dets],"fuente":"YOLO"}

def dibujar_boxes(img,boxes):
    out=img.copy().convert("RGB"); draw=ImageDraw.Draw(out)
    col={"CRITICO":"#e55353","MAYOR":"#f0b429","MENOR":"#3a7bd5","CONFORME":"#27c97e"}
    for (x1,y1,x2,y2),cl,cf in boxes:
        c=col.get(cl,"#fff"); draw.rectangle([x1,y1,x2,y2],outline=c,width=3)
        draw.text((x1+4,y1+4),f"{cl} {cf:.0%}",fill=c)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE VISION
# ══════════════════════════════════════════════════════════════════════════════
MODELOS = ["claude-sonnet-4-6","claude-haiku-4-5-20251001"]

PROMPT_CUERPO_RETRO = """Eres un inspector de control de calidad de conservas enlatadas revisando el resultado de un detector automático (YOLO).
El detector dijo: {clase_yolo} con {conf_yolo:.0%} de confianza.

Analiza la imagen del CUERPO de la lata y determina si estás de acuerdo o no.
Criterios:
- CRITICO: abombamiento, perforación, corrosión profunda, fuga visible, aplastamiento severo.
- MAYOR: abolladuras en sello (borde), deformación estructural, oxidación extendida (>20% superficie).
- MENOR: abolladuras superficiales leves en zona central, raspones sin exponer metal.
- CONFORME: sin defectos o imperfecciones cosméticas mínimas.

Ante la duda, elige la categoría MÁS GRAVE.

Responde SOLO con este JSON sin texto adicional:
{{"clase":"CRITICO|MAYOR|MENOR|CONFORME","confianza":0.00,"descripcion":"breve en español","corrige_yolo":true|false}}"""

PROMPT_CUERPO_SOLO = """Eres un inspector de control de calidad de conservas enlatadas.
Analiza esta imagen del CUERPO de la lata.
- CRITICO: abombamiento, perforación, corrosión profunda, fuga visible.
- MAYOR: abolladuras en sello, deformación estructural, oxidación extendida.
- MENOR: abolladuras leves en zona central, raspones superficiales.
- CONFORME: sin defectos visibles.
Ante la duda elige la MÁS GRAVE.
Responde SOLO con este JSON sin texto adicional:
{"clase":"CRITICO|MAYOR|MENOR|CONFORME","confianza":0.00,"descripcion":"breve en español"}"""

def construir_prompt_etiqueta(fecha_minima_str: str = None) -> str:
    hoy = datetime.now().strftime("%d/%m/%Y")
    anio_hoy = datetime.now().year

    if fecha_minima_str:
        regla_vigente = f"VIGENTE: fecha encontrada e igual o posterior a {fecha_minima_str}."
        regla_vencida = f"VENCIDA: fecha anterior a {fecha_minima_str} O anterior a hoy ({hoy})."
    else:
        regla_vigente = f"VIGENTE: fecha encontrada y posterior a hoy ({hoy})."
        regla_vencida = f"VENCIDA: fecha encontrada y anterior o igual a hoy ({hoy})."

    return (
        "Eres un sistema OCR experto en leer texto grabado/estampado en metal de latas de conserva.\n\n"

        "CONTEXTO: Las fechas en estas latas están grabadas en relieve sobre metal plateado. "
        "Suelen ser pequeñas, pueden estar al revés, rotadas 90/180 grados, o en ángulo. "
        "El formato más común es DDMMMYYYY por ejemplo 18FEB2025 o 31OCT2025. "
        "También aparecen como MMDDYY, MM/YYYY, o YYYY-MM-DD.\n\n"

        "PASO 1 — LOCALIZAR: Busca en TODA la imagen cualquier texto que incluya:\n"
        "  Indicadores: EXP, EXPIRY, EXPR, EXP DATE, EXP., BB, BEST BEFORE, VENCE, VENC, CAD, CONSUME ANTES\n"
        "  Si no encuentras indicador, busca cualquier secuencia de letras+números que parezca fecha.\n\n"

        "PASO 2 — LEER: Rota mentalmente la imagen hasta que el texto tenga sentido. "
        "Descifra letra por letra. Los meses en inglés: JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC. "
        "En español: ENE FEB MAR ABR MAY JUN JUL AGO SEP OCT NOV DIC.\n\n"

        "PASO 3 — CLASIFICAR usando estas reglas ESTRICTAS:\n"
        f"  {regla_vigente}\n"
        f"  {regla_vencida}\n"
        "  ILEGIBLE: texto presente pero ABSOLUTAMENTE imposible leer ningun digito ni letra. "
        "USA ESTO SOLO SI NO PUEDES LEER ABSOLUTAMENTE NADA. "
        f"Si puedes leer al menos el anio (ejemplo: 2025, 2026, 2027), clasifica como VIGENTE o VENCIDA comparando con {anio_hoy}.\n"
        "  SIN_FECHA: la imagen no tiene absolutamente ningun texto visible.\n\n"

        "REGLA DE ORO: Es mejor dar VIGENTE o VENCIDA con confianza baja (0.50) "
        "que dar ILEGIBLE. Solo usa ILEGIBLE si es COMPLETAMENTE imposible.\n\n"

        "Responde SOLO con este JSON, sin texto adicional, sin markdown:\n"
        '{"estado":"VIGENTE|VENCIDA|ILEGIBLE|SIN_FECHA","fecha_leida":"texto exacto leido o null","confianza":0.00,"descripcion":"que texto viste y donde"}'
    )

PROMPT_ETIQUETA = construir_prompt_etiqueta()  # default sin fecha minima

def pil_b64(img,q=92):
    buf=io.BytesIO(); img.convert("RGB").save(buf,format="JPEG",quality=q)
    return base64.standard_b64encode(buf.getvalue()).decode()

def preproc(img,mn=640):
    w,h=img.size
    if min(w,h)<mn:
        f=mn/min(w,h); img=img.resize((int(w*f),int(h*f)),Image.LANCZOS)
    img=ImageEnhance.Contrast(img).enhance(1.4)
    return ImageEnhance.Sharpness(img).enhance(1.8)

def preproc_etiqueta(img, mn=800):
    """
    Preprocesa específicamente imágenes de etiqueta:
    - Mayor resolución (800px)
    - Mayor contraste y nitidez para texto grabado en metal
    - Recorta la zona central donde suele estar la fecha
    """
    w,h=img.size
    # Escalar a resolución mayor
    if min(w,h)<mn:
        f=mn/min(w,h); img=img.resize((int(w*f),int(h*f)),Image.LANCZOS)
    # Recortar zona central (la fecha suele estar en el centro del fondo)
    w2,h2=img.size
    margen_w=int(w2*0.1); margen_h=int(h2*0.1)
    img=img.crop((margen_w, margen_h, w2-margen_w, h2-margen_h))
    # Aumentar contraste y nitidez más agresivamente para texto en metal
    img=ImageEnhance.Contrast(img).enhance(2.0)
    img=ImageEnhance.Sharpness(img).enhance(3.0)
    img=ImageEnhance.Brightness(img).enhance(1.1)
    return img

def parse_json(txt):
    txt=re.sub(r"```[a-z]*","",txt).strip().strip("`").strip()
    try: return json.loads(txt)
    except:
        m=re.search(r'\{[^{}]+\}',txt,re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return None

def llamar_claude(client,b64,prompt,modelo):
    r=client.messages.create(model=modelo,max_tokens=250,messages=[{"role":"user","content":[
        {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":b64}},
        {"type":"text","text":prompt}]}])
    return parse_json(r.content[0].text)

def consenso_claude(client,img,prompt,modelo,intentos=1,es_etiqueta=False):
    """
    Para etiquetas: usa preprocesamiento específico y envía múltiples rotaciones
    si la primera respuesta es ILEGIBLE.
    """
    if es_etiqueta:
        img_proc = preproc_etiqueta(img)
    else:
        img_proc = preproc(img)

    b64=pil_b64(img_proc); res=[]
    for _ in range(intentos):
        try:
            r=llamar_claude(client,b64,prompt,modelo)
            if r: res.append(r)
        except: pass

    # Si es etiqueta y todos los resultados son ILEGIBLE, probar con rotaciones
    if es_etiqueta and all(r.get("estado","") == "ILEGIBLE" for r in res):
        from PIL import Image as _PImage
        for angulo in [90, 180, 270]:
            img_rot = img_proc.rotate(angulo, expand=True)
            b64_rot = pil_b64(img_rot)
            try:
                r = llamar_claude(client, b64_rot, prompt, modelo)
                if r and r.get("estado","") != "ILEGIBLE":
                    res.append(r)
                    break  # Con una buena lectura es suficiente
            except: pass

    if not res: return {"clase":"ERROR","estado":"ERROR","confianza":0.0,"descripcion":"Sin respuesta","fuente":"Claude"}
    campo="clase" if "clase" in res[0] else "estado"
    ganador=Counter(r.get(campo,"ERROR") for r in res).most_common(1)[0][0]
    confs=[r.get("confianza",0.0) for r in res if r.get(campo)==ganador]
    desc=next((r.get("descripcion","") for r in res if r.get(campo)==ganador),"")
    base={campo:ganador,"confianza":round(sum(confs)/len(confs),3),"descripcion":desc,"fuente":"Claude"}
    if campo=="estado":
        fechas=[r.get("fecha_leida") for r in res if r.get(campo)==ganador and r.get("fecha_leida")]
        base["fecha_leida"]=fechas[0] if fechas else None
    return base

def retroalimentar_claude(client,img,res_yolo,modelo):
    """Claude revisa la decisión de YOLO y dice si corrige o confirma."""
    img=preproc(img); b64=pil_b64(img)
    prompt=PROMPT_CUERPO_RETRO.format(clase_yolo=res_yolo["clase"],conf_yolo=res_yolo["confianza"])
    try:
        raw=llamar_claude(client,b64,prompt,modelo)
        if raw:
            raw["fuente"]="Claude+Retro"
            return raw
    except: pass
    return None

def es_nc(rc,re):
    motivos=[]
    if rc.get("clase") in ("CRITICO","MAYOR"): motivos.append(f"Defecto {rc['clase'].lower()}")
    if re.get("estado") in ("VENCIDA","ILEGIBLE","SIN_FECHA"): motivos.append(f"Etiqueta: {re['estado'].lower()}")
    return len(motivos)>0," · ".join(motivos) if motivos else "Conforme"


# ══════════════════════════════════════════════════════════════════════════════
# REPORTE PDF
# ══════════════════════════════════════════════════════════════════════════════
def reporte_pdf(resultados,N,n,c,decision,X,sid,correcciones_count):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas as rlcanvas

    buf=io.BytesIO(); cv=rlcanvas.Canvas(buf,pagesize=A4); W,H=A4
    now=datetime.now(); pp=round(X/n,4) if n>0 else 0

    # P1 — Resumen
    cv.setFillColor(colors.HexColor("#0b1120")); cv.rect(0,0,W,H,fill=1,stroke=0)
    cv.setFillColor(colors.HexColor("#0d1f40")); cv.rect(0,H-90,W,90,fill=1,stroke=0)
    cv.setFillColor(colors.HexColor("#f0b429")); cv.rect(0,H-92,W,3,fill=1,stroke=0)
    cv.setFillColor(colors.white); cv.setFont("Helvetica-Bold",16)
    cv.drawString(1.5*cm,H-38,"REPORTE DE INSPECCIÓN DE LOTE")
    cv.setFont("Helvetica",10)
    cv.drawString(1.5*cm,H-56,"Distribuidora ANCO S.A.S.  ·  ISO 2859-1  ·  NCA 2.5%")
    cv.setFont("Helvetica",9)
    cv.drawString(1.5*cm,H-72,f"Generado: {now.strftime('%d/%m/%Y %H:%M:%S')}   Sesión: {sid}")
    if correcciones_count>0:
        cv.setFillColor(colors.HexColor("#a78bfa"))
        cv.drawString(1.5*cm,H-84,f"Retroalimentacion activa: {correcciones_count} correccion(es) de YOLO por Claude")

    color_dec=colors.HexColor("#27c97e") if decision=="ACEPTADO" else colors.HexColor("#e55353")
    cv.setFillColor(color_dec); cv.roundRect(W-6*cm,H-80,4.5*cm,46,4,fill=1,stroke=0)
    cv.setFillColor(colors.black); cv.setFont("Helvetica-Bold",18)
    cv.drawCentredString(W-3.75*cm,H-52,decision)

    y=H-118
    cv.setFillColor(colors.HexColor("#0d1830")); cv.rect(1.2*cm,y-18,W-2.4*cm,26,fill=1,stroke=0)
    cv.setFillColor(colors.HexColor("#3a7bd5")); cv.rect(1.2*cm,y-18,4,26,fill=1,stroke=0)
    cv.setFont("Helvetica-Bold",9); cv.setFillColor(colors.HexColor("#3a7bd5"))
    cv.drawString(1.8*cm,y-5,f"N={N}  ·  n={n}  ·  c={c}  ·  X={X}  ·  p'={pp:.1%}")

    y-=42; headers=["#","LATA","CUERPO","FUENTE","CONFIANZA","ETIQUETA","DECISIÓN"]
    cw=[1.0*cm,1.8*cm,3.0*cm,2.0*cm,2.5*cm,2.5*cm,3.5*cm]
    cx=[1.2*cm]
    for w_ in cw[:-1]: cx.append(cx[-1]+w_)
    cv.setFillColor(colors.HexColor("#1a3a6a")); cv.rect(1.2*cm,y-6,W-2.4*cm,20,fill=1,stroke=0)
    cv.setFont("Helvetica-Bold",7); cv.setFillColor(colors.HexColor("#7aa3cc"))
    for i,h in enumerate(headers): cv.drawString(cx[i]+3,y+5,h)
    y-=10
    for idx,lata in enumerate(resultados):
        if y<2*cm:
            cv.showPage(); cv.setFillColor(colors.HexColor("#0b1120")); cv.rect(0,0,W,H,fill=1,stroke=0); y=H-2*cm
        bg=colors.HexColor("#0d1830") if idx%2==0 else colors.HexColor("#091020")
        cv.setFillColor(bg); cv.rect(1.2*cm,y-14,W-2.4*cm,18,fill=1,stroke=0)
        nc=lata["no_conforme"]; clase=lata["cuerpo"].get("clase","?")
        estado=lata["etiqueta"].get("estado","?"); conf_c=lata["cuerpo"].get("confianza",0)
        fuente=lata["cuerpo"].get("fuente","?")
        corregido=lata.get("corregido",False)
        dec_txt="NO CONFORME" if nc else ("OBS" if clase=="MENOR" else "CONFORME")
        dec_color=colors.HexColor("#e55353") if nc else (colors.HexColor("#f0b429") if clase=="MENOR" else colors.HexColor("#27c97e"))
        cv.setFont("Helvetica",7); cv.setFillColor(colors.HexColor("#5a7a9a"))
        cv.drawString(cx[0]+3,y-4,str(idx+1))
        cv.setFillColor(colors.white); cv.drawString(cx[1]+3,y-4,lata["id"])
        cv.setFillColor(colors.HexColor("#c8d6e5")); cv.drawString(cx[2]+3,y-4,clase[:10])
        fuente_color=colors.HexColor("#a78bfa") if corregido else colors.HexColor("#7aa3cc")
        cv.setFillColor(fuente_color); cv.drawString(cx[3]+3,y-4,fuente[:8])
        cv.setFillColor(colors.HexColor("#7aa3cc")); cv.drawString(cx[4]+3,y-4,f"{conf_c:.0%}")
        cv.setFillColor(colors.HexColor("#c8d6e5")); cv.drawString(cx[5]+3,y-4,estado[:10])
        cv.setFillColor(dec_color); cv.setFont("Helvetica-Bold",7)
        cv.drawString(cx[6]+3,y-4,dec_txt); y-=18

    # P2 — CCO
    cv.showPage(); cv.setFillColor(colors.HexColor("#0b1120")); cv.rect(0,0,W,H,fill=1,stroke=0)
    cv.setFillColor(colors.HexColor("#0d1f40")); cv.rect(0,H-50,W,50,fill=1,stroke=0)
    cv.setFont("Helvetica-Bold",13); cv.setFillColor(colors.white)
    cv.drawString(1.5*cm,H-30,"CURVA CARACTERÍSTICA DE OPERACIÓN (CCO)")
    cv.setFont("Helvetica",9); cv.setFillColor(colors.HexColor("#5a7a9a"))
    cv.drawString(1.5*cm,H-44,f"Plan: n={n}, c={c}  ·  NCA=2.5%  ·  Binomial")
    ml=2.5*cm; mr=W-1.5*cm; mi=3.5*cm; ag=mr-ml; alt=H-2*cm-mi-70
    ps,pas=calcular_cco(n,c); alpha_v,nql_v=riesgos(n,c)
    def px(p): return ml+p*ag
    def py(pa): return mi+pa*alt+70
    path=cv.beginPath(); path.moveTo(px(0),py(0))
    for p,pa in zip(ps,pas): path.lineTo(px(p),py(pa))
    path.lineTo(px(1),py(0)); path.close()
    cv.setFillColor(colors.HexColor("#0d2040")); cv.drawPath(path,fill=1,stroke=0)
    cv.setStrokeColor(colors.HexColor("#3a7bd5")); cv.setLineWidth(2)
    path2=cv.beginPath(); path2.moveTo(px(ps[0]),py(pas[0]))
    for p,pa in zip(ps[1:],pas[1:]): path2.lineTo(px(p),py(pa))
    cv.drawPath(path2,fill=0,stroke=1)
    cv.setStrokeColor(colors.HexColor("#1a3a6a")); cv.setLineWidth(0.8)
    cv.line(px(0),py(0),px(1),py(0)); cv.line(px(0),py(0),px(0),py(1))
    cv.setFont("Helvetica",7); cv.setFillColor(colors.HexColor("#5a7a9a"))
    for val in [0,0.02,0.05,0.10,0.20,0.30,0.50]:
        cv.line(px(val),py(0)-3,px(val),py(0))
        cv.drawCentredString(px(val),py(0)-12,f"{int(val*100)}%")
    for val in [0,0.2,0.4,0.6,0.8,1.0]:
        cv.line(px(0)-3,py(val),px(0),py(val))
        cv.drawRightString(px(0)-6,py(val)-3,f"{int(val*100)}%")
    cv.setStrokeColor(colors.HexColor("#f0b429")); cv.setLineWidth(1); cv.setDash([4,3])
    cv.line(px(0.025),py(0),px(0.025),py(pa_bin(0.025,n,c))); cv.setDash()
    cv.setFillColor(colors.HexColor("#f0b429")); cv.setFont("Helvetica-Bold",7)
    pa_nca=pa_bin(0.025,n,c)
    cv.drawCentredString(px(0.025),py(0)-22,"NCA 2.5%")
    cv.drawCentredString(px(0.025),py(pa_nca)+8,f"1-a={pa_nca:.0%}")
    if nql_v:
        cv.setStrokeColor(colors.HexColor("#e55353")); cv.setDash([4,3])
        cv.line(px(nql_v),py(0),px(nql_v),py(pa_bin(nql_v,n,c))); cv.setDash()
        cv.setFillColor(colors.HexColor("#e55353"))
        cv.drawCentredString(px(nql_v),py(0)-22,f"NQL {nql_v:.0%}")
    ly=mi+20; cv.setFont("Helvetica",8)
    cv.setFillColor(colors.HexColor("#f0b429"))
    cv.drawString(ml,ly,f"α={alpha_v:.1%} riesgo productor")
    if nql_v:
        cv.setFillColor(colors.HexColor("#e55353"))
        cv.drawString(ml,ly-14,f"β≤10% riesgo consumidor  NQL={nql_v:.0%}")

    # ── Helpers para insertar imágenes PIL en reportlab ──────────────────
    def _pil_to_rl(img_pil, max_w, max_h):
        img_pil = img_pil.convert("RGB")
        img_pil.thumbnail((int(max_w), int(max_h)), Image.LANCZOS)
        tmp = io.BytesIO()
        img_pil.save(tmp, format="JPEG", quality=82)
        tmp.seek(0)
        return tmp, img_pil.size

    # P3 — EVIDENCIA COMPLETA: todas las latas con fotos
    cv.showPage()
    cv.setFillColor(colors.HexColor("#0b1120")); cv.rect(0,0,W,H,fill=1,stroke=0)
    cv.setFillColor(colors.HexColor("#0d1f40")); cv.rect(0,H-50,W,50,fill=1,stroke=0)
    cv.setFont("Helvetica-Bold",13); cv.setFillColor(colors.white)
    cv.drawString(1.5*cm,H-30,f"EVIDENCIA FOTOGRÁFICA – TODAS LAS LATAS ({n} inspeccionadas)")
    cv.setFont("Helvetica",9); cv.setFillColor(colors.HexColor("#5a7a9a"))
    cv.drawString(1.5*cm,H-44,"Cuerpo (izq) · Etiqueta (der) · Verde = Conforme · Rojo = No conforme · Amarillo = Observación")

    # Layout: 2 latas por fila, cada lata ocupa ancho/2
    # Cada bloque: cabecera (20pt) + 2 fotos lado a lado (130pt alto) = ~155pt por lata
    # 2 latas por fila = bloques de 2 columnas
    IMG_W = (W - 3.5*cm) / 2   # ancho de cada foto
    IMG_H = 3.8*cm              # alto de cada foto
    BLOQUE_H = IMG_H + 1.4*cm  # alto total por lata (foto + texto)
    COLS = 2
    MARGEN_IZQ = 1.2*cm

    y = H - 62
    col = 0

    for lata in resultados:
        # Verificar espacio — nueva página si no cabe
        if y - BLOQUE_H < 1.5*cm:
            cv.showPage()
            cv.setFillColor(colors.HexColor("#0b1120")); cv.rect(0,0,W,H,fill=1,stroke=0)
            cv.setFillColor(colors.HexColor("#0d1f40")); cv.rect(0,H-36,W,36,fill=1,stroke=0)
            cv.setFont("Helvetica-Bold",10); cv.setFillColor(colors.white)
            cv.drawString(1.5*cm,H-22,"EVIDENCIA FOTOGRÁFICA (continuación)")
            y = H - 48
            col = 0

        nc = lata["no_conforme"]
        clase = lata["cuerpo"].get("clase","?")
        estado = lata["etiqueta"].get("estado","?")
        conf_c = lata["cuerpo"].get("confianza",0)
        conf_e = lata["etiqueta"].get("confianza",0)

        if nc:
            badge_color = colors.HexColor("#e55353")
            badge_txt = "NO CONFORME"
            bg_lata = colors.HexColor("#1a0707")
        elif clase == "MENOR":
            badge_color = colors.HexColor("#f0b429")
            badge_txt = "OBSERVACION"
            bg_lata = colors.HexColor("#1a1507")
        else:
            badge_color = colors.HexColor("#27c97e")
            badge_txt = "CONFORME"
            bg_lata = colors.HexColor("#071507")

        # Posición X según columna
        x_bloque = MARGEN_IZQ + col * (W - 2*MARGEN_IZQ) / COLS
        ancho_bloque = (W - 2*MARGEN_IZQ) / COLS - 0.3*cm

        # Fondo del bloque
        cv.setFillColor(bg_lata)
        cv.roundRect(x_bloque, y - BLOQUE_H, ancho_bloque, BLOQUE_H, 3, fill=1, stroke=0)
        cv.setStrokeColor(badge_color); cv.setLineWidth(0.8)
        cv.roundRect(x_bloque, y - BLOQUE_H, ancho_bloque, BLOQUE_H, 3, fill=0, stroke=1)

        # Cabecera del bloque
        cv.setFont("Helvetica-Bold", 8)
        cv.setFillColor(badge_color)
        cv.drawString(x_bloque + 4, y - 10, f"#{lata['id']}  {badge_txt}")
        cv.setFont("Helvetica", 6.5)
        cv.setFillColor(colors.HexColor("#c8d6e5"))
        info_txt = f"Cuerpo: {clase} {conf_c:.0%} | Etiqueta: {estado} {conf_e:.0%}"
        cv.drawString(x_bloque + 4, y - 20, info_txt[:55])

        # Fotos: cuerpo a la izquierda, etiqueta a la derecha
        foto_w = (ancho_bloque - 0.4*cm) / 2
        foto_h = IMG_H
        y_foto = y - BLOQUE_H + 0.2*cm

        for foto_idx, (img_key, label) in enumerate([("img_cuerpo_bytes","C"), ("img_etiqueta_bytes","E")]):
            _raw = lata.get(img_key)
            img_pil = Image.open(io.BytesIO(_raw)).convert("RGB") if _raw else None
            x_foto = x_bloque + 0.1*cm + foto_idx * (foto_w + 0.2*cm)
            if img_pil:
                try:
                    tmp_buf, (iw, ih) = _pil_to_rl(img_pil, foto_w * 2.835, foto_h * 2.835)
                    from reportlab.lib.utils import ImageReader
                    rl_img = ImageReader(tmp_buf)
                    # Mantener proporción
                    scale = min(foto_w / (iw / 2.835), foto_h / (ih / 2.835))
                    draw_w = (iw / 2.835) * scale
                    draw_h = (ih / 2.835) * scale
                    x_center = x_foto + (foto_w - draw_w) / 2
                    cv.drawImage(rl_img, x_center, y_foto, width=draw_w, height=draw_h,
                                 preserveAspectRatio=True, mask="auto")
                except Exception:
                    cv.setFillColor(colors.HexColor("#1a2744"))
                    cv.rect(x_foto, y_foto, foto_w, foto_h, fill=1, stroke=0)
                    cv.setFillColor(colors.HexColor("#5a7a9a"))
                    cv.setFont("Helvetica", 6)
                    cv.drawCentredString(x_foto + foto_w/2, y_foto + foto_h/2, f"{label} sin imagen")
            else:
                cv.setFillColor(colors.HexColor("#1a2744"))
                cv.rect(x_foto, y_foto, foto_w, foto_h, fill=1, stroke=0)

        # Avanzar columna o bajar fila
        col += 1
        if col >= COLS:
            col = 0
            y -= BLOQUE_H + 0.3*cm

    # Si quedó columna impar, bajar también
    if col > 0:
        y -= BLOQUE_H + 0.3*cm

    cv.save(); buf.seek(0); return buf

# ══════════════════════════════════════════════════════════════════════════════
# ESTADO
# ══════════════════════════════════════════════════════════════════════════════
for k,v in [("historial_lotes",[]),("resultados_actuales",[]),
            ("session_id",datetime.now().strftime("%Y%m%d%H%M%S"))]:
    if k not in st.session_state: st.session_state[k]=v

def get_key():
    k=os.environ.get("ANTHROPIC_API_KEY","").strip()
    if k: return k
    try: return st.secrets["ANTHROPIC_API_KEY"].strip()
    except: return ""

def make_client(key): return anthropic.Anthropic(api_key=key) if key else None

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙ Configuración")
    api_in=st.text_input("API Key Anthropic",type="password",placeholder="sk-ant-api03-...")
    if api_in: os.environ["ANTHROPIC_API_KEY"]=api_in.strip()

    st.markdown("---")
    modelo_sel=st.selectbox("Modelo Claude",options=MODELOS,
        format_func=lambda m:"Sonnet 4.6 (recomendado)" if "sonnet" in m else "Haiku 4.5 (rápido)")

    intentos=st.select_slider("Votaciones Claude (etiqueta)",options=[1,2,3],value=1,
        help="Solo aplica a la etiqueta cuando YOLO está activo, o a todo cuando YOLO no está.")

    st.markdown("---")
    # ── Retroalimentación ─────────────────────────────────────────────
    st.markdown("**🔄 Retroalimentación YOLO ↔ Claude**")
    retro_activo=st.toggle("Activar retroalimentación",value=False,
        help="Claude Vision revisa cada decisión de YOLO sobre el cuerpo y la corrige si difiere. Genera un log CSV descargable para reentrenamiento.")

    if retro_activo:
        st.markdown(
            '<div style="background:#0d0a1a;border:1px solid #3a2a5a;border-left:3px solid #a78bfa;'
            'border-radius:4px;padding:8px 10px;font-size:11px;font-family:monospace;color:#a78bfa">'
            '● Retroalimentación ON<br>'
            '<span style="color:#6a5a8a">YOLO analiza → Claude revisa → si difiere, Claude gana<br>'
            'Sin almacenamiento adicional</span>'
            '</div>', unsafe_allow_html=True)



    st.markdown("---")
    st.markdown("**📅 Vigencia mínima de etiqueta**")
    usar_fecha_min = st.checkbox("Activar fecha mínima de aceptación", value=False,
        help="Solo se aceptan latas cuya fecha de vencimiento sea posterior a la fecha configurada.")
    if usar_fecha_min:
        col_fm1, col_fm2 = st.columns(2)
        with col_fm1:
            mes_min = st.number_input("Mes", min_value=1, max_value=12, value=1, step=1, key="mes_min")
        with col_fm2:
            anio_min = st.number_input("Año", min_value=2000, max_value=2099, value=datetime.now().year, step=1, key="anio_min")
        try:
            fecha_minima = datetime(int(anio_min), int(mes_min), 1)
            fecha_minima_str = fecha_minima.strftime("%d/%m/%Y")
            st.markdown(
                f'<div style="background:#0d0a1a;border:1px solid #3a2a5a;border-left:3px solid #a78bfa;'
                f'border-radius:4px;padding:8px 10px;font-size:11px;font-family:monospace;margin-top:6px">'
                f'<span style="color:#a78bfa">Aceptar solo si vence después de</span><br>'
                f'<b style="color:#f0b429;font-size:14px">{mes_min:02d}/{int(anio_min)}</b>'
                f'</div>', unsafe_allow_html=True)
        except ValueError:
            st.error("Fecha inválida.", icon="❌")
            fecha_minima_str = None
    else:
        fecha_minima_str = None

    st.session_state["fecha_minima_str"] = fecha_minima_str

    st.markdown("---")
    mostrar_fotos=st.checkbox("Mostrar fotos en resultados",value=True)

    st.markdown("---")
    st.markdown("**Estado del sistema**")
    if YOLO_DISPONIBLE:
        ym=cargar_yolo()
        if ym:
            st.markdown('<div style="background:#071a0f;border:1px solid #1a5a3a;border-left:3px solid #27c97e;border-radius:4px;padding:8px 12px;font-family:monospace;font-size:12px"><span style="color:#27c97e">● YOLO ACTIVO</span><br><span style="color:#4a7a5a">best_latas_defectos.pt · mAP50: 97.1%</span></div>',unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#1a1a07;border:1px solid #5a5a1a;border-left:3px solid #f0b429;border-radius:4px;padding:8px 12px;font-family:monospace;font-size:12px"><span style="color:#f0b429">⚠ YOLO: .pt NO ENCONTRADO</span><br><span style="color:#7a7a4a">Sube best_latas_defectos.pt al repo</span></div>',unsafe_allow_html=True)
    else:
        err_txt = YOLO_ERROR_MSG[:60] if YOLO_ERROR_MSG else "onnxruntime/cv2 no instalado"
        st.markdown(f'<div style="background:#1a0707;border:1px solid #5a1a1a;border-left:3px solid #e55353;border-radius:4px;padding:8px 12px;font-family:monospace;font-size:12px"><span style="color:#e55353">✗ YOLO NO DISPONIBLE</span><br><span style="color:#7a4a4a">{err_txt}<br>→ Claude Vision analiza el cuerpo</span></div>',unsafe_allow_html=True)

    api_check=get_key()
    color_claude="#27c97e" if api_check else "#e55353"
    txt_claude="● CLAUDE ACTIVO" if api_check else "✗ SIN API KEY"
    st.markdown(f'<div style="background:{"#071a0f" if api_check else "#1a0707"};border:1px solid {"#1a5a3a" if api_check else "#5a1a1a"};border-left:3px solid {color_claude};border-radius:4px;padding:8px 12px;font-family:monospace;font-size:12px;margin-top:8px"><span style="color:{color_claude}">{txt_claude}</span></div>',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Historial de lotes**")
    if st.session_state.historial_lotes:
        for lote in reversed(st.session_state.historial_lotes[-8:]):
            cd="#27c97e" if lote["decision"]=="ACEPTADO" else "#e55353"
            st.markdown(f'<div class="hist-item"><span class="hist-id">Lote {lote["id"]}</span><span style="color:{cd}">{lote["decision"]} X={lote["X"]}/{lote["n"]}</span></div>',unsafe_allow_html=True)
        if st.button("🗑 Limpiar historial"):
            st.session_state.historial_lotes=[]; st.session_state.resultados_actuales=[]; st.rerun()
    else:
        st.caption("Sin lotes inspeccionados")

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
retro_txt=" · 🔄 Retroalimentación activa" if retro_activo else ""
st.markdown(
    f'<div class="hero"><div class="hero-title">🥫 SISTEMA DE INSPECCIÓN DE LOTES – ANCO S.A.S.</div>'
    f'<p class="hero-sub">ISO 2859-1 · NCA 2.5% · YOLOv8n (cuerpo) + Claude Vision (etiqueta){retro_txt} · Gerencia y Control de Calidad</p></div>',
    unsafe_allow_html=True)

api_key=get_key(); client=make_client(api_key)
if not api_key: st.warning("Ingresa tu API Key en el panel izquierdo.",icon="🔑")
else: st.success("API Key detectada.",icon="✅")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_plan,tab_insp,tab_hist=st.tabs(["📐 PLAN DE MUESTREO","🔬 INSPECCIÓN","📊 HISTORIAL & CCO"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – PLAN
# ─────────────────────────────────────────────────────────────────────────────
with tab_plan:
    st.markdown("#### Determinación del plan según ISO 2859-1")

    # Modo automático / manual
    modo_plan=st.radio("Modo de plan",["🤖 Automático (ISO 2859-1)","✏️ Manual"],horizontal=True)

    if modo_plan.startswith("🤖"):
        col1,col2=st.columns([1,2])
        with col1:
            N_input=st.number_input("Tamaño del lote N",min_value=2,max_value=500000,value=60,step=1)
        letra,n_auto,c_auto=iso_plan(N_input)
        with col2:
            st.markdown(f'<div class="plan-box"><div class="plan-row"><span><span class="plan-param">LETRA: </span><span class="plan-val">{letra}</span></span><span><span class="plan-param">n = </span><span class="plan-val">{n_auto}</span></span><span><span class="plan-param">ACEPTAR X ≤ </span><span class="plan-val">{c_auto}</span></span><span><span class="plan-param">RECHAZAR X > </span><span class="plan-val">{c_auto}</span></span></div></div>',unsafe_allow_html=True)
        n_plan,c_plan=n_auto,c_auto
    else:
        st.info("En modo manual defines N, n y c libremente. La ISO 2859-1 sugiere los valores a la derecha como referencia.",icon="✏️")
        col_m1,col_m2,col_m3,col_m4=st.columns(4)
        with col_m1: N_input=st.number_input("N – Tamaño del lote",min_value=2,max_value=500000,value=60,step=1,key="N_man")
        letra_ref,n_ref,c_ref=iso_plan(N_input)
        with col_m2: n_plan=st.number_input(f"n – Muestra (ISO sugiere {n_ref})",min_value=1,max_value=N_input,value=n_ref,step=1)
        with col_m3: c_plan=st.number_input(f"c – Aceptación (ISO sugiere {c_ref})",min_value=0,max_value=n_plan,value=c_ref,step=1)
        with col_m4: st.metric("Letra ISO referencia",letra_ref)
        if n_plan!=n_ref or c_plan!=c_ref:
            st.warning(f"⚠️ Estás usando n={n_plan}, c={c_plan} en lugar de n={n_ref}, c={c_ref} recomendados por ISO 2859-1 para N={N_input}.",icon="⚠️")

    alpha_p,nql_p=riesgos(n_plan,c_plan); pa_nca_p=pa_bin(0.025,n_plan,c_plan)
    st.markdown(f'<div class="metrics-row"><div class="met neu"><span class="met-val met-neu">{N_input}</span><div class="met-lbl">Tamaño lote N</div></div><div class="met neu"><span class="met-val met-neu">{n_plan}</span><div class="met-lbl">Muestra n</div></div><div class="met warn"><span class="met-val met-warn">{c_plan}</span><div class="met-lbl">Núm. aceptación c</div></div><div class="met ok"><span class="met-val met-ok">{pa_nca_p:.0%}</span><div class="met-lbl">Pa en NCA (1−α)</div></div></div>',unsafe_allow_html=True)
    st.info(f"**Interpretación:** Inspecciona **{n_plan}** latas. Si X ≤ {c_plan} → ACEPTAS. Si X ≥ {c_plan+1} → RECHAZAS. Pa en NCA = {pa_nca_p:.1%} (α = {alpha_p:.1%}).",icon="📐")

    st.markdown("---")
    col_r1,col_r2=st.columns(2)
    with col_r1:
        st.markdown("**Defectos de cuerpo (YOLOv8n)**")
        st.markdown("| Clase | NC |\n|---|---|\n| CRÍTICO | ✅ Sí |\n| MAYOR | ✅ Sí |\n| MENOR | ⚠️ Obs |\n| CONFORME | ❌ No |")
    with col_r2:
        _fm_activa = st.session_state.get("fecha_minima_str")
        st.markdown("**Defectos de etiqueta (Claude Vision)**")
        if _fm_activa:
            st.markdown(
                f'<div style="background:#0d1a2a;border:1px solid #1a3a5a;border-left:3px solid #f0b429;'
                f'border-radius:4px;padding:8px 12px;font-size:12px;font-family:monospace;margin-bottom:8px">'
                f'📅 Fecha mínima activa: <b style="color:#f0b429">{_fm_activa}</b><br>'
                f'<span style="color:#5a7a9a">Latas que venzan antes de esta fecha → NO CONFORME</span>'
                f'</div>', unsafe_allow_html=True)
        st.markdown("| Estado | NC |\n|---|---|\n| VENCIDA o antes de fecha mín. | ✅ Sí |\n| ILEGIBLE | ✅ Sí |\n| SIN_FECHA | ✅ Sí |\n| VIGENTE (después de fecha mín.) | ❌ No |")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – INSPECCIÓN
# ─────────────────────────────────────────────────────────────────────────────
with tab_insp:
    if not client:
        st.error("Configura la API Key primero.",icon="🔑"); st.stop()

    st.markdown("#### Configuración del lote")
    c1,c2,c3=st.columns(3)
    with c1: N_lote=st.number_input("N – Tamaño del lote",min_value=2,max_value=500000,value=N_input,step=1,key="N_insp")

    # Respetar modo del plan
    if modo_plan.startswith("🤖"):
        _,n_lote,c_lote=iso_plan(N_lote)
    else:
        n_lote=n_plan; c_lote=c_plan

    with c2: st.metric("n – Latas a inspeccionar",n_lote)
    with c3: st.metric("c – Número de aceptación",c_lote)

    st.markdown("---")
    st.markdown(f"#### Subir imágenes <span style='color:#5a7a9a;font-size:13px'>({n_lote} latas · 2 fotos por lata)</span>",unsafe_allow_html=True)
    cu1,cu2=st.columns(2)
    with cu1:
        fotos_c=st.file_uploader(f"📸 CUERPO ({n_lote} fotos)",type=["jpg","jpeg","png","webp","bmp"],accept_multiple_files=True,key="fc")
    with cu2:
        fotos_e=st.file_uploader(f"🏷️ ETIQUETA ({n_lote} fotos)",type=["jpg","jpeg","png","webp","bmp"],accept_multiple_files=True,key="fe")

    if fotos_c or fotos_e:
        nc_u=len(fotos_c) if fotos_c else 0; ne_u=len(fotos_e) if fotos_e else 0; pares=min(nc_u,ne_u)
        if nc_u!=ne_u: st.warning(f"Cuerpos: {nc_u} · Etiquetas: {ne_u} → se analizarán {pares} pares.",icon="⚠️")
        else: st.success(f"{pares} pares listos.",icon="✅")

    puede=client and fotos_c and fotos_e and len(fotos_c)>0 and len(fotos_e)>0

    if st.button("🔬 INICIAR INSPECCIÓN",disabled=not puede,type="primary"):
        fc_s=sorted(fotos_c,key=lambda f:f.name); fe_s=sorted(fotos_e,key=lambda f:f.name)
        total=min(len(fc_s),len(fe_s)); resultados=[]; X=0; obs=0; correcciones=0

        yolo_model=cargar_yolo()
        yolo_ok=yolo_model is not None and YOLO_DISPONIBLE

        prog=st.progress(0,"Iniciando...")
        status=st.empty()

        for i in range(total):
            lid=str(i+1).zfill(3)
            prog.progress(i/total,f"Lata {lid} de {total}...")
            img_c=Image.open(fc_s[i]).convert("RGB")
            img_e=Image.open(fe_s[i]).convert("RGB")
            # Comprimir fotos inmediatamente para liberar memoria RAM
            def _compress(img, max_px=640):
                img.thumbnail((max_px, max_px), Image.LANCZOS)
                buf_=io.BytesIO()
                img.save(buf_, format="JPEG", quality=75)
                buf_.seek(0)
                return buf_.getvalue()
            bytes_c = _compress(img_c)
            # Etiqueta: mayor calidad (85%) para preservar texto grabado
            buf_e_=io.BytesIO(); img_e.save(buf_e_,format="JPEG",quality=85); buf_e_.seek(0)
            bytes_e = buf_e_.getvalue()

            corregido=False; yolo_orig=None

            if yolo_ok:
                # ── YOLO analiza cuerpo ──────────────────────────────────
                status.markdown(f"`[{lid}]` 🤖 YOLO analizando cuerpo...")
                res_c=analizar_yolo(yolo_model,img_c)

                if retro_activo:
                    # ── Claude revisa a YOLO ─────────────────────────────
                    status.markdown(f"`[{lid}]` 👁 Claude revisando decisión de YOLO...")
                    _img_c_retro=Image.open(io.BytesIO(bytes_c)).convert("RGB")
                    retro=retroalimentar_claude(client,_img_c_retro,res_c,modelo_sel)
                    del _img_c_retro
                    if retro:
                        yolo_orig=res_c.copy()
                        corrige=retro.get("corrige_yolo",False) or retro.get("clase")!=res_c["clase"]
                        if corrige:
                            res_c=retro; corregido=True; correcciones+=1
            else:
                # ── Sin YOLO: Claude analiza el cuerpo ───────────────────
                status.markdown(f"`[{lid}]` 👁 Claude analizando cuerpo (YOLO no disponible)...")
                res_c=consenso_claude(client,img_c,PROMPT_CUERPO_SOLO,modelo_sel,intentos)

            # ── Claude analiza etiqueta ──────────────────────────────────
            votos_e=1 if yolo_ok and not retro_activo else intentos
            status.markdown(f"`[{lid}]` 🏷️ Claude analizando etiqueta...")
            _fm = st.session_state.get("fecha_minima_str")
            _prompt_e = construir_prompt_etiqueta(_fm) if _fm else PROMPT_ETIQUETA
            res_e=consenso_claude(client,img_e,_prompt_e,modelo_sel,votos_e,es_etiqueta=True)

            nc,motivo=es_nc(res_c,res_e)
            if nc: X+=1
            elif res_c.get("clase")=="MENOR": obs+=1

            resultados.append({"id":lid,"cuerpo":res_c,"etiqueta":res_e,"no_conforme":nc,"motivo":motivo,
                "img_cuerpo_bytes":bytes_c,"img_etiqueta_bytes":bytes_e,"corregido":corregido,"yolo_original":yolo_orig})
            # Liberar objetos PIL inmediatamente
            del img_c, img_e, bytes_c, bytes_e

        prog.progress(1.0,"✅ Inspección completada."); status.empty()
        decision="ACEPTADO" if X<=c_lote else "RECHAZADO"
        pp=X/total if total>0 else 0

        # No guardamos resultados en session_state para ahorrar RAM
        st.session_state.historial_lotes.append({"id":datetime.now().strftime("%H%M%S"),
            "N":N_lote,"n":total,"c":c_lote,"X":X,"decision":decision})
        # Limitar historial a últimos 10 lotes para no crecer indefinidamente
        if len(st.session_state.historial_lotes) > 10:
            st.session_state.historial_lotes = st.session_state.historial_lotes[-10:]

        # ── Decisión ────────────────────────────────────────────────────
        cls_d="decision-acepta" if decision=="ACEPTADO" else "decision-rechaza"
        cv_="#27c97e" if decision=="ACEPTADO" else "#e55353"
        retro_txt_dec = ("  Correcciones YOLO: " + str(correcciones)) if retro_activo and correcciones>0 else ""
        simbolo_dec = "<=" if decision=="ACEPTADO" else ">"
        sub_txt = f"X = {X} {simbolo_dec} c = {c_lote} | p = {pp:.1%} | {total} latas {retro_txt_dec}"
        st.markdown(f'<div class="decision-box {cls_d}"><div class="decision-title" style="color:{cv_}">{decision}</div><div class="decision-sub">{sub_txt}</div></div>',unsafe_allow_html=True)

        # ── Métricas ────────────────────────────────────────────────────
        cl_x="ok" if decision=="ACEPTADO" else "bad"; cv_x="met-ok" if decision=="ACEPTADO" else "met-bad"
        st.markdown(f'<div class="metrics-row"><div class="met {cl_x}"><span class="met-val {cv_x}">{X}</span><div class="met-lbl">No conformes X</div></div><div class="met neu"><span class="met-val met-neu">{total-X-obs}</span><div class="met-lbl">Conformes</div></div><div class="met warn"><span class="met-val met-warn">{obs}</span><div class="met-lbl">Observaciones</div></div><div class="met neu"><span class="met-val met-neu">{pp:.1%}</span><div class="met-lbl">p\' estimada</div></div></div>',unsafe_allow_html=True)

        if retro_activo and correcciones>0:
            st.markdown(f'<div class="retro-box">🔄 Retroalimentación: Claude corrigió a YOLO en <b style="color:#a78bfa">{correcciones}</b> de {total} latas. El resultado final usa la clasificación de Claude.</div>',unsafe_allow_html=True)

        # ── Detalle por lata ─────────────────────────────────────────────
        st.markdown("---"); st.markdown("#### Detalle por lata")
        for lata in resultados:
            nc=lata["no_conforme"]; clase=lata["cuerpo"].get("clase","?")
            estado=lata["etiqueta"].get("estado","?")
            conf_c=lata["cuerpo"].get("confianza",0); conf_e=lata["etiqueta"].get("confianza",0)
            fuente=lata["cuerpo"].get("fuente","?")
            corr=lata.get("corregido",False); yo=lata.get("yolo_original")

            badge=f'<span class="badge badge-nc">NO CONFORME</span>' if nc else (f'<span class="badge badge-obs">OBSERVACIÓN</span>' if clase=="MENOR" else f'<span class="badge badge-ok">CONFORME</span>')
            corr_tag=f'<span class="corr-tag">⚡ CORREGIDO POR CLAUDE</span>' if corr else ""
            yo_txt=f" <span style='color:#6a5a8a;font-size:11px'>[YOLO decía: {yo['clase']} {yo['confianza']:.0%}]</span>" if yo else ""
            detalle=(f"[{fuente}] Cuerpo: <b>{clase}</b> ({conf_c:.0%}){yo_txt} – {lata['cuerpo'].get('descripcion','')[:55]} | "
                     f"[Claude] Etiqueta: <b>{estado}</b> ({conf_e:.0%})"
                     +(f" · {lata['etiqueta'].get('fecha_leida','')}" if lata["etiqueta"].get("fecha_leida") else ""))

            st.markdown(f'<div class="lata-card"><div class="lata-id">#{lata["id"]}</div><div class="lata-detail">{detalle}</div>{corr_tag}{badge}</div>',unsafe_allow_html=True)

            if mostrar_fotos and nc:
                ci1,ci2=st.columns(2)
                with ci1:
                    _img_c=Image.open(io.BytesIO(lata["img_cuerpo_bytes"])).convert("RGB")
                    boxes=lata["cuerpo"].get("boxes",[])
                    img_ann=dibujar_boxes(_img_c,boxes) if boxes else _img_c
                    lbl=f"Cuerpo {lata['id']} ({fuente})"
                    if corr: lbl+=" corregido"
                    st.image(img_ann,caption=lbl,width=220)
                    del _img_c, img_ann
                with ci2:
                    _img_e=Image.open(io.BytesIO(lata["img_etiqueta_bytes"])).convert("RGB")
                    st.image(_img_e,caption=f"Etiqueta {lata['id']} (Claude)",width=220)
                    del _img_e

        # ── PDF ────────────────────────────────────────────────────────
        st.markdown("---")
        pdf=reporte_pdf(resultados,N_lote,total,c_lote,decision,X,st.session_state.session_id,correcciones)
        st.download_button("📄 Descargar Reporte PDF",data=pdf,
            file_name=f"reporte_ANCO_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",mime="application/pdf")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – HISTORIAL & CCO
# ─────────────────────────────────────────────────────────────────────────────
with tab_hist:
    st.markdown("#### Curva Característica de Operación (CCO)")
    if st.session_state.historial_lotes:
        ult=st.session_state.historial_lotes[-1]; n_c=ult["n"]; c_c=ult["c"]
    else:
        _,n_c,c_c=iso_plan(60)
    ps,pas=calcular_cco(n_c,c_c); alpha_v,nql_v=riesgos(n_c,c_c)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.ticker as ticker
        import warnings; warnings.filterwarnings("ignore")
        fig,ax=plt.subplots(figsize=(9,4.5))
        fig.patch.set_facecolor("#0b1120"); ax.set_facecolor("#0d1830")
        ax.fill_between(ps,pas,alpha=0.15,color="#3a7bd5")
        ax.plot(ps,pas,color="#3a7bd5",linewidth=2.5,label=f"Pa(p)  n={n_c}, c={c_c}")
        ax.axhline(0.10,color="#e55353",linestyle="--",linewidth=1,alpha=0.6,label="β=10%")
        ax.axvline(0.025,color="#f0b429",linestyle="--",linewidth=1,alpha=0.8,label="NCA=2.5%")
        pa_nca_v=pa_bin(0.025,n_c,c_c)
        ax.annotate(f"1−α={pa_nca_v:.0%}",xy=(0.025,pa_nca_v),xytext=(0.07,pa_nca_v-0.12),
            arrowprops=dict(arrowstyle="->",color="#f0b429"),color="#f0b429",fontsize=8)
        if nql_v:
            ax.axvline(nql_v,color="#e55353",linestyle=":",linewidth=1,alpha=0.8,label=f"NQL≈{nql_v:.0%}")
            ax.annotate(f"β≤10%\nNQL={nql_v:.0%}",xy=(nql_v,0.10),xytext=(nql_v+0.02,0.25),
                arrowprops=dict(arrowstyle="->",color="#e55353"),color="#e55353",fontsize=8)
        if st.session_state.historial_lotes:
            ult=st.session_state.historial_lotes[-1]
            pp_=ult["X"]/ult["n"] if ult["n"]>0 else 0
            ax.scatter([pp_],[pa_bin(pp_,n_c,c_c)],
                color="#27c97e" if ult["decision"]=="ACEPTADO" else "#e55353",
                zorder=5,s=80,label=f"Último lote p'={pp_:.1%}")
        ax.set_xlabel("Proporción real de defectos p",color="#7aa3cc",fontsize=10)
        ax.set_ylabel("Pa(p)",color="#7aa3cc",fontsize=10)
        ax.set_title(f"CCO · n={n_c}, c={c_c} · ISO 2859-1 · NCA=2.5%",color="#c8d6e5",fontsize=11,pad=10)
        ax.tick_params(colors="#5a7a9a"); ax.spines[:].set_color("#1a3a6a")
        ax.set_xlim(0,0.5); ax.set_ylim(0,1.05)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.grid(True,color="#1a2744",linewidth=0.5,alpha=0.7)
        ax.legend(loc="upper right",facecolor="#0d1830",edgecolor="#1a3a6a",labelcolor="#c8d6e5",fontsize=8)
        st.pyplot(fig); plt.close(fig)
    except ImportError:
        st.warning("Instala matplotlib para ver la CCO.",icon="⚠️")

    nql_s=f"{nql_v:.0%}" if nql_v else "N/D"
    st.markdown(f'<div class="metrics-row"><div class="met warn"><span class="met-val met-warn">{alpha_v:.1%}</span><div class="met-lbl">α Riesgo productor</div></div><div class="met bad"><span class="met-val met-bad">≤10%</span><div class="met-lbl">β Riesgo consumidor</div></div><div class="met ok"><span class="met-val met-ok">{pa_bin(0.025,n_c,c_c):.0%}</span><div class="met-lbl">Pa en NCA</div></div><div class="met neu"><span class="met-val met-neu">{nql_s}</span><div class="met-lbl">NQL estimado</div></div></div>',unsafe_allow_html=True)

    if st.session_state.historial_lotes:
        st.markdown("---"); st.markdown("#### Historial de lotes")
        st.dataframe([{"Lote":l["id"],"N":l["N"],"n":l["n"],"c":l["c"],"X":l["X"],
            "p'":f"{l['X']/l['n']:.1%}" if l["n"]>0 else "—","Decisión":l["decision"]}
            for l in reversed(st.session_state.historial_lotes)])

st.markdown('<div class="footer">ANCO S.A.S. · ISO 2859-1 · NCA 2.5% · YOLOv8n + Claude Vision · Gerencia y Control de Calidad · UNICAUCA 2026</div>',unsafe_allow_html=True)
