import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import io
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Generador de Nube de Palabras",
    page_icon="⚜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS NOIR GÓTICOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,300;0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"], .stApp {
        background-color: #050608 !important;
        color: #e5e7eb !important;
        font-family: 'Spectral', Georgia, serif !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        background-color: #08090d !important;
        border-right: 1px solid rgba(212, 175, 55, 0.18) !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #d4af37 !important;
        font-family: 'Spectral', serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.05em !important;
    }

    textarea, input[type="text"] {
        background-color: #0b0c12 !important;
        border: 1px solid rgba(212, 175, 55, 0.25) !important;
        border-radius: 4px !important;
        color: #f3f4f6 !important;
        font-family: 'Spectral', serif !important;
        font-size: 0.95rem !important;
    }

    .stButton > button {
        background: #141724 !important;
        color: #d4af37 !important;
        border: 1px solid rgba(212, 175, 55, 0.4) !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
    }
    .stButton > button:hover {
        background: #1e2235 !important;
        border-color: #d4af37 !important;
        box-shadow: 0 0 15px rgba(212, 175, 55, 0.2) !important;
    }

    [data-testid="metric-container"] {
        background: #090a0f;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-top: 2px solid #d4af37;
        border-radius: 4px;
        padding: 14px;
    }

    .header-card {
        background: #090b10;
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-left: 4px solid #d4af37;
        border-radius: 4px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }

    .freq-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 12px;
        margin: 4px 0;
        background: #090a0f;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }
    .freq-bar {
        height: 5px;
        background: #d4af37;
        border-radius: 2px;
        display: inline-block;
    }
    .rank-tag {
        background: rgba(212, 175, 55, 0.1);
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-radius: 3px;
        padding: 1px 6px;
        font-size: 0.72rem;
        color: #d4af37;
        font-family: 'JetBrains Mono', monospace;
    }
    .wc-container {
        background: #000000;
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-radius: 4px;
        padding: 12px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PALABRAS COMUNES Y PALETAS
# ─────────────────────────────────────────────
STOPWORDS_ES = {
    "de","la","el","en","y","a","los","del","se","las","un","por","con","no","una","su",
    "para","es","al","lo","como","mas","pero","sus","le","ya","o","este","si","porque",
    "esta","entre","cuando","muy","sin","sobre","tambien","me","hasta","hay","donde",
    "quien","desde","nos","durante","ni","contra","ese","eso","ante","bajo","tras",
    "que","fue","son","han","ha","ser","era","estan","siendo","sido","he","has","hemos",
    "habian","tiene","tienen","hacer","puede","pueden","asi","tan","parte","todo","todos",
    "todas","cada","otro","otra","otros","otras","mismo","misma","nuestro","nuestra",
    "ellos","ellas","nosotros","les","esa","esos","esas","aquel","aquella","aquellos",
}

def obtener_stopwords(idioma):
    sw = set(STOPWORDS)
    if idioma in ("Español", "Ambos"):
        sw |= STOPWORDS_ES
    return sw

PALETAS = {
    "Dorado Clásico":     ["#d4af37", "#f59e0b", "#c5a059", "#8b7355", "#e5c07b", "#fffbeb"],
    "Plata y Gris":       ["#ffffff", "#e5e7eb", "#9ca3af", "#6b7280", "#4b5563", "#374151"],
    "Azul Acero":         ["#38bdf8", "#0284c7", "#7dd3fc", "#0369a1", "#bae6fd", "#e0f2fe"],
    "Rojo Cálido":        ["#f43f5e", "#be123c", "#fb7185", "#881337", "#fda4af", "#ffe4e6"],
    "Monocromático":      ["#fafafa", "#d4d4d8", "#a1a1aa", "#71717a", "#52525b", "#3f3f46"],
}

FORMAS = {
    "Rectángulo": None,
    "Círculo": "circle",
}

def crear_mascara(forma, size=500):
    if forma == "circle":
        y, x = np.ogrid[:size, :size]
        cx, cy = size // 2, size // 2
        mascara = np.ones((size, size), dtype=np.uint8) * 255
        mascara[(x - cx)**2 + (y - cy)**2 <= (size // 2 - 12)**2] = 0
        return mascara
    return None

def limpiar_texto(texto, stopwords, min_longitud):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-záéíóúüñàâèêîôùûäëïöü\s]", " ", texto, flags=re.UNICODE)
    palabras = [p for p in texto.split() if p not in stopwords and len(p) >= min_longitud]
    return " ".join(palabras)

def contar_palabras(texto_limpio):
    return pd.DataFrame(Counter(texto_limpio.split()).most_common(50), columns=["Palabra", "Frecuencia"])

def generar_wordcloud(texto_limpio, paleta_nombre, max_words, fondo, forma, ancho=1000, alto=520):
    import random
    colores = PALETAS[paleta_nombre]

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        rng = random_state or random.Random()
        return colores[rng.randint(0, len(colores) - 1)]

    mascara = crear_mascara(forma, size=min(ancho, alto))
    wc = WordCloud(
        width=ancho, height=alto, max_words=max_words,
        background_color=fondo, color_func=color_func,
        mask=mascara, collocations=False,
        min_font_size=11, max_font_size=120,
        prefer_horizontal=0.8, relative_scaling=0.5, margin=5,
    ).generate(texto_limpio)

    fig, ax = plt.subplots(figsize=(ancho / 100, alto / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor(fondo)
    plt.tight_layout(pad=0)
    return fig

def fig_a_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# PANEL LATERAL
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Nube de Palabras")
    st.divider()

    st.markdown("### 1. Texto de origen")
    fuente = st.radio("fuente", ["Escribir / Pegar", "Subir archivo"], label_visibility="collapsed")
    texto_input = ""

    if fuente == "Escribir / Pegar":
        texto_input = st.text_area("Texto a analizar:", height=170, placeholder="Pega aquí un artículo, opiniones o cualquier documento...")

        with st.expander("Ver textos de ejemplo"):
            ejemplos = {
                "Inteligencia Artificial": """
                La inteligencia artificial es una rama de la informática que desarrolla sistemas
                capaces de resolver tareas que requieren habilidades cognitivas. El aprendizaje automático,
                las redes neuronales y el procesamiento de lenguaje natural son sus pilares técnicos.
                Los modelos de gran escala y la visión por computadora se aplican hoy en salud, educación,
                finanzas y logística con resultados medibles.
                """,
                "Colombia": """
                Colombia es un país ubicado al noroccidente de América del Sur, destacado por su biodiversidad,
                riqueza cultural y variedad geográfica. Bogotá es su capital y principal centro económico,
                seguida por Medellín, Cali y Barranquilla. El café colombiano tiene reconocimiento mundial
                por su calidad, y el país cuenta con ecosistemas en los Andes, la Amazonía, el Caribe y el Pacífico.
                """,
                "Industria 4.0": """
                La cuarta revolución industrial moderniza los modelos de producción mediante la integración
                de internet de las cosas, analítica de datos, inteligencia artificial y automatización.
                Las fábricas conectadas utilizan sensores y software en tiempo real para optimizar procesos,
                mientras la computación en la nube asegura la continuidad de las operaciones.
                """,
            }
            ejemplo_sel = st.selectbox("Elegir ejemplo:", list(ejemplos.keys()), label_visibility="collapsed")
            if st.button("Cargar este texto"):
                st.session_state["texto_ejemplo"] = ejemplos[ejemplo_sel]
                st.rerun()

        if "texto_ejemplo" in st.session_state and not texto_input:
            texto_input = st.session_state["texto_ejemplo"]
    else:
        archivo = st.file_uploader("Archivo de texto:", type=["txt", "csv"], label_visibility="collapsed")
        if archivo:
            if archivo.name.endswith(".txt"):
                texto_input = archivo.read().decode("utf-8", errors="ignore")
            elif archivo.name.endswith(".csv"):
                df_csv = pd.read_csv(archivo)
                col_txt = st.selectbox("Columna con el texto:", df_csv.columns.tolist())
                texto_input = " ".join(df_csv[col_txt].dropna().astype(str).tolist())
            st.success(f"Archivo listo: {len(texto_input):,} caracteres.")

    st.divider()
    st.markdown("### 2. Filtros")
    idioma = st.selectbox("Eliminar palabras comunes:", ["Español", "Inglés", "Ambos", "Ninguno"])
    min_longitud = st.slider("Letras mínimas por palabra:", 2, 8, 3)
    palabras_extra = st.text_input("Omitir palabras específicas:", placeholder="ejemplo: además, también")

    st.divider()
    st.markdown("### 3. Aspecto visual")
    paleta_sel = st.selectbox("Gama de color:", list(PALETAS.keys()))
    fondo_sel = st.radio("Fondo:", ["Negro", "Blanco"], horizontal=True)
    fondo_color = "black" if fondo_sel == "Negro" else "white"
    forma_sel = st.selectbox("Forma exterior:", list(FORMAS.keys()))
    max_words = st.slider("Límite de palabras:", 20, 200, 80)

    st.divider()
    generar = st.button("Crear nube de palabras", use_container_width=True)


# ─────────────────────────────────────────────
# CONTENIDO PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1 style="margin:0; font-family:'Spectral'; font-size:1.8rem; color:#ffffff;">Generador de Nube de Palabras</h1>
    <p style="margin:2px 0 0 0; color:#d4af37 !important; font-family:'JetBrains Mono'; font-size:0.78rem;">
        Identifica rápidamente los conceptos centrales de cualquier documento
    </p>
</div>
""", unsafe_allow_html=True)

if not generar or not texto_input.strip():
    col_izq, col_der = st.columns([3, 2], gap="large")
    with col_izq:
        st.markdown("""
        ### Cómo funciona
        Las palabras con mayor frecuencia aparecen con un tamaño de letra más grande.
        Es una forma visual e intuitiva de entender de qué habla un texto sin leerlo por completo.
        """)
        st.markdown("- **Filtro automático:** Descarta artículos y conectores como *el, de, en, con*.")
        st.markdown("- **Personalizable:** Cambia la paleta de color, forma y cantidad de palabras.")
        st.markdown("- **Descargable:** Guarda la imagen en alta definición o la tabla en CSV.")
    with col_der:
        st.markdown("### Usos habituales")
        st.markdown("- Resúmenes de encuestas y opiniones de clientes.")
        st.markdown("- Análisis de noticias, discursos y artículos.")
        st.markdown("- Revisión de temáticas clave en tesis o reportes.")
    st.stop()

# ─────────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────────
stopwords_set = obtener_stopwords(idioma) if idioma != "Ninguno" else set()
if palabras_extra.strip():
    stopwords_set |= {p.strip().lower() for p in palabras_extra.split(",") if p.strip()}

texto_limpio = limpiar_texto(texto_input, stopwords_set, min_longitud)
if not texto_limpio.strip():
    st.error("No quedaron palabras válidas tras aplicar los filtros. Prueba bajando la longitud mínima.")
    st.stop()

df_freq = contar_palabras(texto_limpio)
total_palabras = len(texto_limpio.split())
vocabulario = len(df_freq)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Palabras procesadas", f"{total_palabras:,}")
m2.metric("Palabras distintas", f"{vocabulario:,}")
m3.metric("Palabra principal", df_freq.iloc[0]["Palabra"] if not df_freq.empty else "—")
m4.metric("Apariciones", int(df_freq.iloc[0]["Frecuencia"]) if not df_freq.empty else 0)

st.write("")

with st.spinner("Creando imagen..."):
    fig_wc = generar_wordcloud(texto_limpio, paleta_sel, max_words, fondo_color, FORMAS[forma_sel], ancho=1000, alto=520)

st.markdown('<div class="wc-container">', unsafe_allow_html=True)
st.pyplot(fig_wc, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

img_bytes = fig_a_bytes(fig_wc)
st.download_button("Descargar imagen (PNG)", data=img_bytes, file_name="nube_palabras.png", mime="image/png", use_container_width=True)

st.divider()

col_freq, col_tabla = st.columns([3, 2], gap="large")
with col_freq:
    st.markdown("### Las 20 palabras más repetidas")
    top20 = df_freq.head(20)
    max_freq = top20["Frecuencia"].max()

    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        p = row["Palabra"]
        f = int(row["Frecuencia"])
        barra_w = max(10, int((f / max_freq) * 200))
        st.markdown(
            f'<div class="freq-row">'
            f'<span class="rank-tag">#{rank:02d}</span>'
            f'<span style="font-weight:600; color:#f3f4f6; min-width:130px; font-size:0.95rem;">{p}</span>'
            f'<div class="freq-bar" style="width:{barra_w}px;"></div>'
            f'<span style="font-family:\'JetBrains Mono\'; font-size:0.85rem; color:#d4af37; min-width:30px; text-align:right;">{f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

with col_tabla:
    st.markdown("### Tabla detallada")
    st.dataframe(df_freq.head(30), use_container_width=True, height=420)
    csv_bytes = df_freq.to_csv(index=False).encode("utf-8")
    st.download_button("Exportar datos (CSV)", data=csv_bytes, file_name="frecuencias.csv", mime="text/csv", use_container_width=True)

plt.close("all")
