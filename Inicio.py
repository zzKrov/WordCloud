import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import io
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(
    page_title="Generador de Nube de Palabras // Krov Industries",
    page_icon="⚜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# ESTILOS VISUALES Y RESTRICCIÓN DE ESCALA
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;0,700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"], .stApp {
        background-color: #050608 !important;
        color: #e5e7eb !important;
        font-family: 'Spectral', Georgia, serif !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .brand-signature {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        color: #9ca3af;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    [data-testid="stSidebar"] {
        background-color: #08090d !important;
        border-right: 1px solid rgba(212, 175, 55, 0.18) !important;
    }

    .header-card {
        background: #090b10;
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-left: 4px solid #d4af37;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }

    /* Contenedor acotado y centrado para la nube */
    .wc-container {
        max-width: 820px;
        margin: 0 auto 16px auto;
        background: #000000;
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 6px;
        padding: 8px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.7);
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .freq-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 5px 10px;
        margin: 3px 0;
        background: #090a0f;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }
    .freq-bar {
        height: 4px;
        background: #d4af37;
        border-radius: 2px;
        display: inline-block;
    }
    .rank-tag {
        background: rgba(212, 175, 55, 0.1);
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-radius: 3px;
        padding: 1px 5px;
        font-size: 0.7rem;
        color: #d4af37;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

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
    "Dorado Clásico":     ["#d4af37", "#f59e0b", "#c5a059", "#8b7355", "#e5c07b"],
    "Plata y Gris":       ["#ffffff", "#e5e7eb", "#9ca3af", "#6b7280", "#4b5563"],
    "Azul Acero":         ["#38bdf8", "#0284c7", "#7dd3fc", "#0369a1", "#bae6fd"],
    "Rojo Cálido":        ["#f43f5e", "#be123c", "#fb7185", "#881337", "#fda4af"],
    "Monocromático":      ["#fafafa", "#d4d4d8", "#a1a1aa", "#71717a", "#52525b"],
}

FORMAS = {
    "Rectángulo": None,
    "Círculo": "circle",
}

def crear_mascara(forma, size=400):
    if forma == "circle":
        y, x = np.ogrid[:size, :size]
        cx, cy = size // 2, size // 2
        mascara = np.ones((size, size), dtype=np.uint8) * 255
        mascara[(x - cx)**2 + (y - cy)**2 <= (size // 2 - 10)**2] = 0
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

def generar_wordcloud(texto_limpio, paleta_nombre, max_words, fondo, forma):
    import random
    colores = PALETAS[paleta_nombre]

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        rng = random_state or random.Random()
        return colores[rng.randint(0, len(colores) - 1)]

    ancho, alto = 800, 380
    mascara = crear_mascara(forma, size=alto) if forma == "circle" else None
    wc = WordCloud(
        width=ancho, height=alto, max_words=max_words,
        background_color=fondo, color_func=color_func,
        mask=mascara, collocations=False,
        min_font_size=10, max_font_size=85,
        prefer_horizontal=0.82, relative_scaling=0.45, margin=4,
    ).generate(texto_limpio)

    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=130)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor(fondo)
    plt.tight_layout(pad=0)
    return fig

# ---------------------------------------------------------
# BARRA LATERAL
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-signature">Krov Industries · Ajustes</div>', unsafe_allow_html=True)
    st.markdown("### Configuración de Texto")

    fuente = st.radio("Origen:", ["Escribir / Pegar", "Subir archivo"], label_visibility="collapsed")
    texto_input = ""

    if fuente == "Escribir / Pegar":
        texto_input = st.text_area("Texto a analizar:", height=150, placeholder="Pega aquí el contenido...")
        with st.expander("Ver ejemplos preparados"):
            ejemplos = {
                "Inteligencia Artificial": "La inteligencia artificial y el aprendizaje automático optimizan procesos complejos mediante redes neuronales profundas y análisis de datos en tiempo real.",
                "Industria 4.0": "La manufactura avanzada integra internet de las cosas, analítica predictiva y automatización para incrementar la eficiencia operativa en fábricas modernas.",
            }
            sel_ex = st.selectbox("Ejemplo:", list(ejemplos.keys()), label_visibility="collapsed")
            if st.button("Cargar ejemplo", use_container_width=True):
                st.session_state["texto_ejemplo"] = ejemplos[sel_ex]
                st.rerun()

        if "texto_ejemplo" in st.session_state and not texto_input:
            texto_input = st.session_state["texto_ejemplo"]
    else:
        archivo = st.file_uploader("Archivo (.txt, .csv):", type=["txt", "csv"], label_visibility="collapsed")
        if archivo:
            if archivo.name.endswith(".txt"):
                texto_input = archivo.read().decode("utf-8", errors="ignore")
            elif archivo.name.endswith(".csv"):
                df_csv = pd.read_csv(archivo)
                col_txt = st.selectbox("Columna de texto:", df_csv.columns.tolist())
                texto_input = " ".join(df_csv[col_txt].dropna().astype(str).tolist())

    st.markdown("### Filtros")
    idioma = st.selectbox("Descartar palabras comunes:", ["Español", "Inglés", "Ambos", "Ninguno"])
    min_longitud = st.slider("Letras mínimas:", 2, 8, 3)
    palabras_extra = st.text_input("Omitir términos puntuales:", placeholder="ejemplo: además, entonces")

    st.markdown("### Estilo de Nube")
    paleta_sel = st.selectbox("Paleta:", list(PALETAS.keys()))
    fondo_sel = st.radio("Fondo:", ["Negro", "Blanco"], horizontal=True)
    fondo_color = "black" if fondo_sel == "Negro" else "white"
    forma_sel = st.selectbox("Contorno:", list(FORMAS.keys()))
    max_words = st.slider("Límite de palabras:", 20, 150, 70)

    generar = st.button("Generar nube de palabras", use_container_width=True)

# ---------------------------------------------------------
# CONTENIDO PRINCIPAL
# ---------------------------------------------------------
st.markdown("""
<div class="header-card">
    <div class="brand-signature">Krov Industries · Analítica Semántica</div>
    <h2 style="margin:0; font-family:'Spectral'; font-size:1.6rem; color:#ffffff;">Generador de Nube de Palabras</h2>
    <p style="margin:2px 0 0 0; color:#d4af37 !important; font-family:'JetBrains Mono'; font-size:0.75rem;">
        Identificación visual proporcionada de conceptos y términos dominantes
    </p>
</div>
""", unsafe_allow_html=True)

if not generar or not texto_input.strip():
    c_info1, c_info2 = st.columns([1.2, 1], gap="medium")
    with c_info1:
        st.markdown("""
        ### Modo de uso
        Introduce o carga un documento desde el panel lateral y presiona **Generar nube de palabras**.
        El sistema procesará las frecuencias léxicas y construirá una representación gráfica adaptada a las proporciones de lectura en pantalla.
        """)
    with c_info2:
        st.info("Configura la exclusión de conectores y la paleta de color para obtener una salida visualmente equilibrada.")
    st.stop()

stopwords_set = obtener_stopwords(idioma) if idioma != "Ninguno" else set()
if palabras_extra.strip():
    stopwords_set |= {p.strip().lower() for p in palabras_extra.split(",") if p.strip()}

texto_limpio = limpiar_texto(texto_input, stopwords_set, min_longitud)
if not texto_limpio.strip():
    st.error("No se encontraron palabras suficientes tras aplicar los filtros. Intenta reducir la longitud mínima.")
    st.stop()

df_freq = contar_palabras(texto_limpio)
total_palabras = len(texto_limpio.split())
vocabulario = len(df_freq)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Palabras evaluadas", f"{total_palabras:,}")
m2.metric("Vocabulario único", f"{vocabulario:,}")
m3.metric("Término líder", df_freq.iloc[0]["Palabra"] if not df_freq.empty else "—")
m4.metric("Frecuencia máxima", int(df_freq.iloc[0]["Frecuencia"]) if not df_freq.empty else 0)

st.write("")
with st.spinner("Construyendo visualización..."):
    fig_wc = generar_wordcloud(texto_limpio, paleta_sel, max_words, fondo_color, FORMAS[forma_sel])

st.markdown('<div class="wc-container">', unsafe_allow_html=True)
st.pyplot(fig_wc, use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

buf = io.BytesIO()
fig_wc.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=fig_wc.get_facecolor())
st.download_button(
    "Descargar nube en alta resolución (PNG)",
    data=buf.getvalue(),
    file_name="nube_de_palabras.png",
    mime="image/png",
    use_container_width=True
)

st.divider()
c_freq, c_tab = st.columns([1.2, 1], gap="large")
with c_freq:
    st.markdown("### Frecuencias destacadas")
    top15 = df_freq.head(15)
    max_f = top15["Frecuencia"].max()
    for rank, (_, row) in enumerate(top15.iterrows(), 1):
        p = row["Palabra"]
        f = int(row["Frecuencia"])
        b_w = max(10, int((f / max_f) * 160))
        st.markdown(
            f'<div class="freq-row">'
            f'<span class="rank-tag">#{rank:02d}</span>'
            f'<span style="font-weight:600; color:#f3f4f6; min-width:110px; font-size:0.9rem;">{p}</span>'
            f'<div class="freq-bar" style="width:{b_w}px;"></div>'
            f'<span style="font-family:\'JetBrains Mono\'; font-size:0.8rem; color:#d4af37; min-width:30px; text-align:right;">{f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

with c_tab:
    st.markdown("### Tabla exportable")
    st.dataframe(df_freq.head(25), use_container_width=True, height=340)
    st.download_button(
        "Exportar frecuencias (CSV)",
        data=df_freq.to_csv(index=False).encode("utf-8"),
        file_name="frecuencias_palabras.csv",
        mime="text/csv",
        use_container_width=True
    )

plt.close("all")
