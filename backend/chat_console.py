import os
import re
import logging
import pandas as pd
from pathlib import Path

from hugchat import hugchat
from sentence_transformers import SentenceTransformer, util
from pydantic_settings import BaseSettings

from rapidfuzz import process, fuzz  # para corrección de typos

logging.basicConfig(level=logging.INFO)

# ── CONFIG ────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    CSV_PATH: str          = os.getenv("CSV_PATH", "D:/Repositorios/Malligasta/datos_turisticos.csv")
    COOKIE_PATH: str       = os.getenv("COOKIE_PATH", "D:/Repositorios/Malligasta/backend/cookies.json")
    EMBED_MODEL_NAME: str  = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
    TOP_K: int             = int(os.getenv("TOP_K", 3))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# ── FUNCIONES DE PREPROCESADO ─────────────────────────────────────────
def limpiar_texto(texto: str) -> str:
    if not isinstance(texto, str) or not texto.strip():
        return "SIN_DATA"
    t = texto.lower().strip()
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'[^\w\s,.:/-]', '', t)
    return t

def load_csv() -> pd.DataFrame:
    df = pd.read_csv(settings.CSV_PATH)
    # limpiamos todas las columnas
    for col in df.columns:
        df[col] = df[col].astype(str).apply(lambda x: limpiar_texto(x) if x.strip() else "SIN_DATA")
    logging.info("CSV cargado con %d registros.", len(df))
    return df

_df = load_csv()

# ── LISTA DE NOMBRES PARA DETECTAR ENTIDAD ─────────────────────────────
# Asumimos que 'nombre' ya está limpio
NOMBRES = _df["nombre"].unique().tolist()

def detectar_nombre(texto: str):
    """Devuelve el nombre más parecido si score>=80, sino None."""
    match, score, _ = process.extractOne(texto, NOMBRES, scorer=fuzz.ratio)
    return match if score >= 80 else None

# ── PLANTILLAS DINÁMICAS PARA RAG ─────────────────────────────────────
plantillas_base = {
    "ubicación":             "¿Dónde se ubica {nombre}?",
    "telefono":              "¿Cuál es el número de teléfono de {nombre}?",
    "pagina_web":            "¿Cuál es la página web oficial de {nombre}?",
    "instagram":             "¿Cuál es el Instagram de {nombre}?",
    "facebook":              "¿Cuál es el Facebook de {nombre}?",
    "servicios_precios":     "¿Cuál es el rango de precios de {nombre}?",
    "servicios_horarios":    "¿Cuáles son los horarios de {nombre}?",
    "servicios_delivery":    "¿Ofrecen servicio a domicilio {nombre}?",
    "servicios_medios_pago": "¿Qué medios de pago aceptan {nombre}?",
    "agenda":                "Cuéntame la agenda o eventos de {nombre}.",
    "tipo_de_comida":        "¿Qué tipo de comida ofrece {nombre}?",
    "foto":                  "¿Podrías mostrarme una foto de {nombre}?",
}

campos = {}
for col in _df.columns:
    if col == "nombre":
        continue
    if col in plantillas_base:
        campos[col] = (plantillas_base[col], col)
    else:
        campos[col] = (f"¿Cuál es el valor de {col} para {{nombre}}?", col)

COLUMN_KEYS = list(campos.keys())

def construir_corpus(df: pd.DataFrame) -> list[str]:
    corpus = []
    for _, row in df.iterrows():
        nombre = row["nombre"]
        if nombre == "sin_data":
            continue
        for plantilla, columna in campos.values():
            valor = row[columna]
            if valor and valor != "sin_data":
                pregunta = plantilla.format(nombre=nombre)
                corpus.append(f"Pregunta: {pregunta}\nRespuesta: {valor}")
    logging.info("Corpus construido con %d fragmentos.", len(corpus))
    return corpus

def embed_corpus(corpus: list[str], model: SentenceTransformer):
    emb = model.encode(corpus, convert_to_tensor=True)
    logging.info("Embeddings del corpus listos.")
    return emb

# ── CORRECCIÓN DE TYPOS ────────────────────────────────────────────────
def corregir_typos(texto: str) -> str:
    palabras = texto.split()
    corregidas = []
    for w in palabras:
        match, score, _ = process.extractOne(w, COLUMN_KEYS, scorer=fuzz.ratio)
        corregidas.append(match if score >= 75 else w)
    return " ".join(corregidas)

# ── RAG + HuggingChat ─────────────────────────────────────────────────
_embed_model = SentenceTransformer(settings.EMBED_MODEL_NAME)
_corpus       = construir_corpus(_df)
_corpus_emb   = embed_corpus(_corpus, _embed_model)

_chatbot = None
def _get_chatbot():
    global _chatbot
    if _chatbot is None:
        if not Path(settings.COOKIE_PATH).exists():
            raise FileNotFoundError(f"No se encontró cookies en '{settings.COOKIE_PATH}'")
        _chatbot = hugchat.ChatBot(cookie_path=settings.COOKIE_PATH)
        cid = _chatbot.new_conversation()
        _chatbot.change_conversation(cid)
        logging.info("HuggingChat inicializado.")
    return _chatbot

# ── HISTORIAL DE CONVERSACIÓN ─────────────────────────────────────────
_historial     = []
MAX_HISTORIAL  = 5

def rag_chat(pregunta: str) -> str:
    if not pregunta.strip():
        return "Por favor ingresa una pregunta."

    global _historial

    # 1) corregir typos en términos clave
    pregunta_corr = corregir_typos(pregunta.lower())

    # 2) detectar si mencionan un nombre específico
    nombre = detectar_nombre(pregunta_corr)
    if nombre:
        # recupero TODO el registro
        row = _df[_df["nombre"] == nombre].iloc[0]
        # construyo un contexto con **todas** las columnas
        contexto = "\n".join(
            f"{col}: {row[col]}"
            for col in _df.columns
            if row[col] and row[col] != "sin_data"
        )
        prompt = (
            "Eres un asistente turístico amigable y servicial. Responde usando exclusivamente esta información:\n\n"
            f"{contexto}\n\n"
            f"Usuario preguntó por {nombre}: {pregunta_corr}\nAsistente:"
        )
        bot = _get_chatbot()
        respuesta = bot.chat(prompt).text.strip()
        # guardo en historial
        _historial.append({"usuario": pregunta_corr, "asistente": respuesta})
        if len(_historial) > MAX_HISTORIAL:
            _historial = _historial[-MAX_HISTORIAL:]
        return respuesta

    # 3) si no hay nombre, seguimos con RAG por fragmentos
    q_emb      = _embed_model.encode(pregunta_corr, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, _corpus_emb)[0]
    topk      = cos_scores.topk(settings.TOP_K)
    indices   = topk.indices.tolist()
    contexto  = "\n---\n".join(_corpus[i] for i in indices)

    historial_txt = "\n".join(
        f"Usuario: {t['usuario']}\nAsistente: {t['asistente']}"
        for t in _historial[-MAX_HISTORIAL:]
    )

    prompt = (
        "Eres un asistente turístico profesional y servicial. Puedes saludar, preguntar si necesitan más ayuda y usar un lenguaje natural.\n"
        "Usa SOLO la información proporcionada si se hace una pregunta concreta.\n\n"
        f"{contexto}\n\n"
        f"{historial_txt}\n"
        f"Usuario: {pregunta_corr}\nAsistente:"
    )

    bot = _get_chatbot()
    respuesta = bot.chat(prompt).text.strip()
    _historial.append({"usuario": pregunta_corr, "asistente": respuesta})
    if len(_historial) > MAX_HISTORIAL:
        _historial = _historial[-MAX_HISTORIAL:]
    return respuesta
