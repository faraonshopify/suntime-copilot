import os
import unicodedata
from typing import Dict, Any, List

from fastapi import FastAPI
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

# OpenAI SDK (>=1.0)
from openai import OpenAI
from openai._exceptions import OpenAIError

load_dotenv()

app = FastAPI(title="Suntime Inbox Copilot")

# ========= ENV =========
SHOPIFY_DOMAIN = os.getenv("SHOPIFY_DOMAIN")  # ej: suntimestores.myshopify.com
SHOPIFY_TOKEN  = os.getenv("SHOPIFY_ACCESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_VERSION    = "2024-10"

SHOPIFY_GQL = f"https://{SHOPIFY_DOMAIN}/admin/api/{API_VERSION}/graphql.json" if SHOPIFY_DOMAIN else None
STOREFRONT_BASE = "https://suntimestore.com"  # fallback si no hay onlineStoreUrl

# Cliente OpenAI (si hay key)
oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ========= MODELOS =========
class ChatInput(BaseModel):
    message: str

# ========= UTILS =========
def env_ok() -> Dict[str, Any]:
    return {
        "SHOPIFY_DOMAIN_set": bool(SHOPIFY_DOMAIN),
        "SHOPIFY_TOKEN_set":  bool(SHOPIFY_TOKEN),
        "OPENAI_API_KEY_set": bool(OPENAI_API_KEY),
        "SHOPIFY_GQL": SHOPIFY_GQL,
    }

# --- Helpers de texto ---
STOPWORDS_ES = {
    "busco","buscar","quisiera","quiero","deseo","me","para","de","del","la","el","los","las",
    "un","una","unos","unas","por","con","color","talla","modelo","hombre","mujer","niño","niña",
    "porfa","porfavor","favor","tipo","marca","modelo","reloj","lentes","gafas"
}

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def normalize_terms(text: str) -> List[str]:
    """Toma la frase del cliente y devuelve tokens útiles para buscar en Shopify."""
    t = strip_accents(text.lower())
    tokens = [w for w in t.replace(",", " ").replace(".", " ").split() if w]
    useful = []
    for w in tokens:
        if w in STOPWORDS_ES:    # filtra palabras vacías comunes
            continue
        if len(w) <= 2:          # filtra palabras muy cortas
            continue
        useful.append(w)
    # limita a 5 para que la query no sea enorme
    return useful[:5]

BRAND_HINTS = {"invicta","ray-ban","rayban","oakley","new","era","g-shock","gshock","festina","tommy","hilfiger","vans","polaroid"}

def build_shopify_query(terms: List[str]) -> str:
    """
    Construye una query compatible con Shopify:
      - Para cada término, busca en title/vendor/tag con OR
      - Combina los grupos con AND para refinar
      - Usa comodín * para permitir coincidencias parciales
    Ejemplo final: (title:invicta* OR vendor:invicta OR tag:invicta) AND (title:dorado* OR tag:dorado)
    """
    if not terms:
        return ""
    parts = []
    i = 0
    while i < len(terms):
        w = terms[i]
        # juntar "new era", "ray ban", "g shock", "tommy hilfiger"
        pair = (w + " " + terms[i+1]) if i+1 < len(terms) else None
        if pair and pair in {"ray ban","new era","g shock","tommy hilfiger"}:
            w = pair
            i += 1
        w = w.replace('"', '')
        # si parece marca, también probamos en vendor
        group = [
            f'title:"{w}*"',
            f'tag:"{w}"'
        ]
        if any(tok in BRAND_HINTS for tok in w.split()):
            group.append(f'vendor:"{w}"')
        parts.append("(" + " OR ".join(group) + ")")
        i += 1
    return " AND ".join(parts)

def shopify_query(q: str) -> List[Dict[str, Any]]:
    if not SHOPIFY_GQL or not SHOPIFY_TOKEN:
        raise RuntimeError("Shopify no configurado (dominio o token faltante).")

    terms = normalize_terms(q)
    # si quedaron muy pocos términos, al menos usa la frase limpia
    if not terms:
        terms = normalize_terms(strip_accents(q))

    gql_query = build_shopify_query(terms)

    query = """
    query($q:String!){
      products(first: 3, query: $q){
        edges{
          node{
            title
            handle
            vendor
            productType
            onlineStoreUrl
            variants(first:1){edges{node{price}}}
          }
        }
      }
    }
    """
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": SHOPIFY_TOKEN
    }
    resp = requests.post(
        SHOPIFY_GQL,
        json={"query": query, "variables": {"q": gql_query}},
        headers=headers,
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data.get("data", {}).get("products", {}).get("edges", []) or []

def format_price(v: str | float | None) -> str:
    try:
        return f"S/ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "S/ —"

def format_products(edges: List[Dict[str, Any]]) -> str:
    lines = []
    for e in edges[:3]:
        n = e["node"]
        title = n.get("title", "Producto")
        handle = n.get("handle", "")
        url = n.get("onlineStoreUrl") or (f"{STOREFRONT_BASE}/products/{handle}" if handle else STOREFRONT_BASE)

        price = "—"
        try:
            price = n["variants"]["edges"][0]["node"]["price"]
        except Exception:
            pass

        lines.append(f"• {title} — {format_price(price)}\n  {url}")
    return "\n\n".join(lines)

def intent_from_openai(user_text: str) -> str:
    """
    Pide a la IA una consulta corta tipo: 'invicta reloj dorado'.
    Si falla OpenAI, devolvemos el propio texto del usuario.
    """
    if not oai:
        return user_text
    try:
        prompt = (
            "Convierte este mensaje del cliente en una consulta breve para buscar en Shopify "
            "(3-6 palabras, marca+tipo+atributo/color si aplica). "
            f"Texto: {user_text!r}. "
            "Ejemplos: 'invicta reloj dorado', 'ray-ban mujer polarizado', 'new era gorra negra'."
        )
        r = oai.responses.create(model="gpt-4.1-mini", input=prompt)
        intent = (r.output_text or "").strip().lower()
        return intent or user_text
    except OpenAIError:
        return user_text
    except Exception:
        return user_text

# ========= ENDPOINTS =========
@app.get("/")
def root():
    return {"message": "🚀 Copiloto IA de Suntime Store activo"}

@app.get("/health")
def health():
    return {"ok": True, "env": env_ok()}

@app.post("/chat-intent")
def chat_intent(data: ChatInput):
    # 1) normaliza el texto del usuario
    user_text = (data.message or "").strip()
    if not user_text:
        return {"response": "¿Qué estás buscando? Ejemplo: 'reloj Invicta dorado' o 'lentes Ray-Ban mujer polarizado'."}

    # 2) intención (con fallback)
    intent = intent_from_openai(user_text)

    # 3) buscar productos en Shopify
    try:
        products = shopify_query(intent)
    except Exception as e:
        return {"response": f"⚠️ No pude consultar Shopify ({e}). Verifica dominio/token y vuelve a intentar."}

    if not products:
        # prueba una segunda pasada con términos directos del usuario
        try:
            products = shopify_query(user_text)
        except Exception:
            pass

    if not products:
        return {"response": f"No encontré resultados para “{intent}”. ¿Buscamos otra marca o estilo parecido?"}

    # 4) formatear respuesta
    listado = format_products(products)
    return {
  "response": (
    f"✨ Esto es lo más cercano a lo que buscas ({intent}):\n\n"
    f"{listado}\n\n"
    "¿Quieres que te recomiende según precio o color? "
    "🎁 *Tip:* si es tu primera compra, usa **SUNTIME10%** para 10% de descuento."
  )
}