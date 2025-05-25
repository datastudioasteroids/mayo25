import os
import re
import logging
import hashlib
import pandas as pd
from pathlib import Path
from pydantic_settings import BaseSettings
from hugchat import hugchat
from sentence_transformers import SentenceTransformer, util

# ── CONFIG ────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    CSV_PATH: str       = os.getenv("CSV_PATH", "D:/Repositorios/Malligasta/datos_turisticos.csv")
    COOKIE_PATH: str    = os.getenv("COOKIE_PATH", "D:/Repositorios/Malligasta/backend/cookies.json")
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
    TOP_K: int          = int(os.getenv("TOP_K", "3"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Rutas para checksum y modelo entrenado
CSV_HASH_FILE: Path   = Path(settings.CSV_PATH).with_suffix(".md5")
MODEL_SAVE_PATH: Path = Path("modelo_entrenado.bin")

logging.basicConfig(level=logging.INFO)

# ── UTILIDADES ──────────────────────────────────────────────────────
def compute_md5(file_path: str) -> str:
    """Calcula el MD5 de un archivo (usado para detectar cambios en el CSV)."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def limpiar_texto(texto: str) -> str:
    if not isinstance(texto, str) or not texto.strip():
        return "SIN_DATA"
    texto = texto.lower().strip()
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'[^\w\s,.:/-]', '', texto)
    return texto

def load_csv() -> pd.DataFrame:
    """Carga y limpia el CSV completo."""
    df = pd.read_csv(settings.CSV_PATH)
    # Migrado a DataFrame.map para evitar el warning
    df = df.astype(str).applymap(lambda x: limpiar_texto(x) if x.strip() else "SIN_DATA")
    logging.info("CSV cargado con %d registros.", len(df))
    return df

# ── PLANTILLAS PARA RAG ──────────────────────────────────────────────
campos = {
    "ubicacion":      ("¿Dónde se ubica {nombre}?",      "ubicación"),
    "precio":         ("¿Cuál es el rango de precios de {nombre}?", "servicios_precios"),
    "tipo_de_comida": ("¿Qué tipo de comida ofrece {nombre}?",    "tipo_de_comida"),
    "horarios":       ("¿Cuáles son los horarios de {nombre}?",   "servicios_horarios"),
    "agenda":         ("Cuéntame la agenda o eventos de {nombre}.","agenda"),
    "telefono":       ("¿Cuál es el número de teléfono de {nombre}?", "contacto_telefono"),
    "email":          ("¿Cuál es la dirección de correo electrónico de {nombre}?", "contacto_email"),
    "sitio_web":      ("¿Cuál es el sitio web oficial de {nombre}?",   "contacto_web"),
}

def construir_corpus(df: pd.DataFrame, campos: dict) -> list[str]:
    corpus = []
    for _, row in df.iterrows():
        nombre = limpiar_texto(row.get("nombre", ""))
        if nombre == "sin_data":
            continue
        for plantilla, columna in campos.values():
            valor = row.get(columna, None)
            if pd.notnull(valor) and limpiar_texto(str(valor)) != "sin_data":
                pregunta  = plantilla.format(nombre=nombre)
                respuesta = limpiar_texto(str(valor))
                corpus.append(f"Pregunta: {pregunta}\nRespuesta: {respuesta}")
    logging.info("Corpus construido con %d fragmentos.", len(corpus))
    return corpus

# Alias para compatibilidad
def construir_qa_pairs(df, campos):
    return construir_corpus(df, campos)

# ── PREPARE EMBEDDINGS ───────────────────────────────────────────────
_df                 = load_csv()
_corpus             = construir_corpus(_df, campos)
_embed_model        = SentenceTransformer(settings.EMBED_MODEL_NAME)
_corpus_embeddings  = _embed_model.encode(_corpus, convert_to_tensor=True)
logging.info("Embeddings del corpus listos.")

# ── LAZY INIT DE HUGGINGCHAT ────────────────────────────────────────
_chatbot_instance = None

def _get_chatbot():
    global _chatbot_instance
    if _chatbot_instance is None:
        if not os.path.exists(settings.COOKIE_PATH):
            raise FileNotFoundError(f"No se encontró cookies en '{settings.COOKIE_PATH}'.")
        _chatbot_instance = hugchat.ChatBot(cookie_path=settings.COOKIE_PATH)
        cid = _chatbot_instance.new_conversation()
        _chatbot_instance.change_conversation(cid)
        logging.info("Chatbot HuggingChat inicializado.")
    return _chatbot_instance

def chat(query: str, top_k: int = None) -> str:
    """RAG + LLM: tu API/front llama a esta función."""
    if not query.strip():
        return "Por favor ingresa una pregunta."
    top_k = top_k or settings.TOP_K
    q_emb = _embed_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, _corpus_embeddings)[0]
    top_idxs = cos_scores.topk(top_k).indices.tolist()
    contexto = "\n---\n".join(_corpus[i] for i in top_idxs)
    prompt = (
        "Usa SOLO esta info extraída del CSV para responder:\n\n"
        f"{contexto}\n\nPregunta: {query}\nRespuesta:"
    )
    return _get_chatbot().chat(prompt)

# ── STUBS ───────────────────────────────────────────────────────────
def generar_ejemplos(df, campos):
    raise NotImplementedError

def entrenar_modelo():
    raise NotImplementedError

if __name__ == "__main__":
    # Consola de prueba
    while True:
        q = input("> ")
        if q.lower() in ("q", "quit", "salir"):
            break
        print(chat(q))