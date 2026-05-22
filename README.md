# 🥫 Inspección de Lotes · ANCO S.A.S. · Talento Tech 2026

Sistema de inspección de lotes de conservas enlatadas basado en **Claude Vision** + **ISO 2859-1**.  
Detecta defectos físicos y valida fechas de vencimiento sin necesidad de modelo YOLO.

## Pipeline de precisión

```
Foto del cuerpo  →  Claude Vision (clasifica defecto: CRÍTICO / MAYOR / MENOR / CONFORME)
Foto de etiqueta →  Claude Vision (lee y valida fecha de vencimiento)
       ↓
ISO 2859-1: n y c según NCA 2.5% · Decisión ACEPTADO / RECHAZADO
       ↓
Reporte PDF con CCO, evidencia y lista de no conformes
```

Este enfoque supera al OCR tradicional (~90%+ de precisión) gracias al **triple consenso**:  
cada lata se analiza 3 veces y se toma la respuesta más frecuente.

## Características

- ✅ Sin YOLO — solo Claude Vision
- ✅ Plan de muestreo automático según ISO 2859-1 (NCA 2.5%, Nivel II)
- ✅ Triple votación por lata para máxima precisión
- ✅ CCO (Curva Característica de Operación) con α y β
- ✅ Reporte PDF técnico descargable por lote
- ✅ Historial de lotes inspeccionados

## Configuración local

1. Clona el repositorio
2. Instala dependencias:

```bash
pip install -r requirements.txt
```

3. Exporta tu API Key de Anthropic:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

4. Ejecuta la app:

```bash
streamlit run app.py
```

## Despliegue en Streamlit Cloud

1. Haz fork o sube este repositorio a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Selecciona el repo, rama `main`, archivo `app.py`
4. En **Advanced settings → Secrets**, agrega:

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-..."
```

5. Haz clic en **Deploy**

> No se necesita `best.pt` ni ningún modelo local. Todo corre en la nube vía API de Anthropic.

## Costo estimado por lote

| Modelo | Costo por lata (3 votaciones) | Lote de 80 latas |
|---|---|---|
| Haiku 4.5 | ~$0.010 USD | ~$0.80 USD |
| Sonnet 4.6 | ~$0.029 USD | ~$2.30 USD |

## Estructura del repositorio

```
app-ia/
├── app.py            ← Aplicación principal Streamlit
├── requirements.txt  ← Dependencias Python
├── runtime.txt       ← Versión de Python (3.10)
└── README.md         ← Este archivo
```

## Contexto académico

Proyecto integrador — **Gerencia y Control de Calidad**  
Universidad del Cauca · Talento Tech 2026  
Caso: Distribuidora ANCO S.A.S. · Popayán, Colombia
