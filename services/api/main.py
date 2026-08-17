import os
import requests
import psycopg2
import psycopg2.extras
from fastapi import FastAPI

app = FastAPI(title="FM26 Assistant Manager API")

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "dbname": os.environ.get("POSTGRES_DB", "fm26"),
    "user": os.environ.get("POSTGRES_USER", "fm26"),
    "password": os.environ["POSTGRES_PASSWORD"],
}

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def call_ollama(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/rotation-candidates")
def rotation_candidates(limit: int = 8):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT name, position, age, squad_status, appearances,
               sub_appearances, current_ability, potential_ability,
               wage_monthly, contract_expiry
        FROM players
        WHERE export_id = (SELECT export_id FROM players ORDER BY exported_at DESC LIMIT 1)
        ORDER BY COALESCE(appearances, 0) ASC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return {"error": "Ingen spillerdata fundet. Har du kørt en eksport endnu?"}

    player_lines = "\n".join(
        f"- {r['name']} ({r['position']}, {r['age']} år): "
        f"{r['appearances'] or 0} kampe ({r['sub_appearances'] or 0} som indskifter), "
        f"status: {r['squad_status']}, evne {r['current_ability']}/potentiale {r['potential_ability']}, "
        f"kontrakt udløber {r['contract_expiry']}"
        for r in rows
    )

    prompt = f"""Du er en assistant manager i Football Manager 26 for Pafos FC.
Her er de {len(rows)} spillere i truppen med mindst spilletid lige nu:

{player_lines}

Giv en kort, konkret vurdering på dansk af hvilke af disse spillere der bør prioriteres til mere spilletid, rotation, eller et udlån for at holde deres tilfredshed oppe. Vær specifik om hvem og hvorfor, baseret på deres alder, potentiale og kontraktsituation."""

    answer = call_ollama(prompt)
    return {"players_considered": rows, "advice": answer}
