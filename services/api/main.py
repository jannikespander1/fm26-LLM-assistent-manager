import json
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

SYSTEM_PROMPT = (
    "You are an analytical assistant manager for a football club in "
    "Football Manager 26. You give a holistic assessment of each player - "
    "not just whether they should play more. Low appearances alone does "
    "NOT automatically mean a player deserves more minutes. Weigh age, "
    "ability ceiling, and squad standing together, not in isolation."
)

ASSESSMENTS = [
    "Core player", "Promising talent", "Solid squad player",
    "Underused - deserves a chance", "Below required standard",
    "Exit candidate (loan or sale)",
]

ALL_ATTRIBUTES = [
    ("Indlæg", "Crossing"), ("Markering", "Marking"), ("Afslutning", "Finishing"),
    ("Aflevering", "Passing"), ("Tackling", "Tackling"), ("Langskud", "Long Shots"),
    ("Teknik", "Technique"), ("Hovedstød", "Heading"), ("Dribling", "Dribbling"),
    ("Acceleration", "Acceleration"), ("Hurtighed", "Pace"), ("Styrke", "Strength"),
    ("Selvbeherskelse", "Composure"), ("Koncentration", "Concentration"),
    ("Beslutninger", "Decisions"), ("Beslutsomhed", "Determination"), ("Flair", "Flair"),
    ("Lederskab", "Leadership"), ("Bevægelse uden bold", "Off the Ball"),
    ("Positionering", "Positioning"), ("Teamwork", "Teamwork"), ("Overblik", "Vision"),
    ("Arbejdsrate", "Work Rate"), ("Førsteberøring", "First Touch"),
    ("Reflekser", "Reflexes"), ("Håndtering", "Handling"), ("Kontrol af feltet", "Command of Area"),
    ("Kommunikation", "Communication"), ("Rækkevidde i luften", "Aerial Reach"),
    ("Én-mod-énere", "One on Ones"), ("Komme ud (tendens)", "Rushing Out"),
    ("Spark", "Kicking"), ("Boksning", "Punching"), ("Kast", "Throwing"),
    ("Excentricitet", "Eccentricity"),
]

POSITION_KEY_ATTRIBUTES = {
    "GK": ["Reflekser", "Håndtering", "Kontrol af feltet", "Kommunikation", "Rækkevidde i luften",
           "Positionering", "Beslutninger", "Koncentration", "Én-mod-énere", "Komme ud (tendens)",
           "Spark", "Boksning"],
    "DEF": ["Markering", "Tackling", "Positionering", "Hovedstød", "Styrke", "Koncentration",
            "Selvbeherskelse", "Hurtighed", "Beslutninger", "Aflevering"],
    "MID": ["Aflevering", "Overblik", "Beslutninger", "Teamwork", "Arbejdsrate", "Positionering",
            "Teknik", "Bevægelse uden bold", "Tackling", "Selvbeherskelse"],
    "OMID": ["Overblik", "Flair", "Aflevering", "Dribling", "Bevægelse uden bold", "Teknik",
             "Beslutninger", "Indlæg", "Hurtighed", "Acceleration", "Førsteberøring"],
    "ATT": ["Afslutning", "Selvbeherskelse", "Bevægelse uden bold", "Dribling", "Hurtighed",
            "Acceleration", "Flair", "Teknik", "Indlæg"],
}

MATCH_CONTEXT = [
    ("Kampens spiller", "Man of the Match count"),
    ("Progressive afleveringer per 90", "Progressive Passes/90"),
    ("Nøgleafleveringer", "Key Passes"),
    ("Boldbesiddelse vundet per 90", "Possession Won/90"),
    ("Højreben", "Right Foot"), ("Venstreben", "Left Foot"),
    ("Tilfredshed", "Happiness"),
    ("Tilfredshed med spilletid", "Playing Time Satisfaction"),
    ("Missede kampe i træk", "Consecutive matches NOT played"),
    ("Klar til at blive fritstillet", "Club marked as ready to release"),
]

PAIRED_STATS = [
    ("Passing accuracy", "Procentdel af succesfulde afleveringer", "Gennemførte afleveringer", "Afleveringer forsøgt", "pass"),
    ("Shot conversion", "Scor. %", "Mål", "Skud", "shot_conversion"),
    ("Shot accuracy", "Procentdel af skud på mål", "Shots on Target", "Skud", "shot_accuracy"),
    ("Tackle success", "Procentdel af succesfulde tacklinger", "Gennemførte tacklinger", "Tacklinger forsøgt", "tackle"),
    ("Aerial duels won", "Procentdel af vundne hovedstød", "Vundne hovedstødsdueller", "Hovedstødsforsøg", "aerial"),
]

EXPECTATION_PAIRS = [
    ("Finishing vs xG", "xG", "Mål"),
    ("Creativity vs xA", "xA", "Assists"),
]

BENCHMARKS = {
    "pass": [(65, "poor for professional level"), (75, "below average"), (85, "solid/good"), (999, "excellent, elite-level")],
    "shot_conversion": [(8, "poor finishing"), (12, "average"), (18, "good"), (999, "excellent, elite finishing")],
    "shot_accuracy": [(30, "below average"), (40, "average"), (50, "good"), (999, "excellent")],
    "tackle": [(50, "poor - timing/positioning issues"), (60, "below average"), (70, "solid, above average"), (999, "excellent/elite")],
    "aerial": [(45, "poor in the air"), (55, "below average"), (65, "average for the position"), (999, "strong to elite")],
}

MIN_SAMPLE_SIZE = 5  # kamp-statistikker med under dette antal forsøg ignoreres helt


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def try_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in ("", "-"):
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def parse_percent(v):
    if v in (None, ""):
        return None
    try:
        return float(str(v).replace("%", "").replace(",", "."))
    except ValueError:
        return None


def benchmark_tier(category, pct):
    if pct is None:
        return None, None
    for i, (upper, label) in enumerate(BENCHMARKS[category]):
        if pct < upper:
            return i, label
    return 3, "excellent"


def classify_position(pos):
    if not pos:
        return "MID"
    primary = pos.split(",")[0].strip()
    if primary.startswith("Mm"):
        return "GK"
    if primary.startswith("An"):
        return "ATT"
    if primary.startswith("OM"):
        return "OMID"
    if primary.startswith("DM"):
        return "MID"
    if primary.startswith("M"):
        paren = ""
        if "(" in primary and ")" in primary:
            paren = primary[primary.find("(") + 1:primary.find(")")]
        if "V" in paren or "H" in paren:
            return "OMID"
        return "MID"
    if primary.startswith("F"):
        return "DEF"
    return "MID"


def squad_averages_by_position(conn, export_id):
    cur = conn.cursor()
    cur.execute("SELECT position, attributes FROM players WHERE export_id = %s", (export_id,))
    all_rows = cur.fetchall()
    cur.close()

    grouped = {"GK": [], "DEF": [], "MID": [], "OMID": [], "ATT": []}
    for pos, attrs in all_rows:
        grouped[classify_position(pos)].append(attrs or {})

    averages = {}
    for cat, players_attrs in grouped.items():
        cat_avgs = {}
        for dk, _ in ALL_ATTRIBUTES:
            values = [try_float(a.get(dk)) for a in players_attrs]
            values = [v for v in values if v is not None]
            cat_avgs[dk] = round(sum(values) / len(values), 1) if values else None
        averages[cat] = cat_avgs
    return averages


def call_ollama(user_prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.5, "top_p": 0.9, "repeat_penalty": 1.3},
            "keep_alive": "2m",
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def unload_model():
    try:
        requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "keep_alive": 0},
            timeout=30,
        )
    except requests.RequestException:
        pass


def format_all_paired_stats(extras):
    """Kamp-statistikker med for få forsøg (< MIN_SAMPLE_SIZE) ignoreres helt - ikke bare nedvægtet."""
    lines = []
    for label, pct_key, num_key, den_key, category in PAIRED_STATS:
        pct = parse_percent(extras.get(pct_key))
        num, den = extras.get(num_key), try_float(extras.get(den_key))
        if pct is None or not den or den < MIN_SAMPLE_SIZE:
            continue
        _, verdict = benchmark_tier(category, pct)
        lines.append(f"{label}: {pct}% ({num}/{int(den)} attempted) - {verdict}")
    for label, exp_key, actual_key in EXPECTATION_PAIRS:
        exp, actual = try_float(extras.get(exp_key)), try_float(extras.get(actual_key))
        if exp is None or actual is None:
            continue
        diff = actual - exp
        perf = "overperforming" if diff > 0.5 else "underperforming" if diff < -0.5 else "in line with expectation"
        lines.append(f"{label}: {actual} actual vs {exp} expected ({perf})")
    return lines


def pick_standout_attribute(extras, cat_avgs, category):
    relevant_keys = set(POSITION_KEY_ATTRIBUTES[category])
    best = None
    for dk, en in ALL_ATTRIBUTES:
        if dk not in relevant_keys:
            continue
        val = try_float(extras.get(dk))
        avg = cat_avgs.get(dk)
        if val is None or avg is None:
            continue
        dev = abs(val - avg)
        if best is None or dev > best[0]:
            best = (dev, en, val, avg)
    if not best:
        return "No key attribute to present"
    dev, en, val, avg = best
    direction = "above" if val >= avg else "below"
    return f"{en}={val} ({dev:.1f} {direction} {category} avg {avg})"


def pick_standout_context(extras):
    """Ignorerer small samples helt. Er der intet tilbage, sendes en klar besked frem for at gætte."""
    candidates = []
    for label, pct_key, num_key, den_key, category in PAIRED_STATS:
        pct = parse_percent(extras.get(pct_key))
        num, den = extras.get(num_key), try_float(extras.get(den_key))
        if pct is None or not den or den < MIN_SAMPLE_SIZE:
            continue
        tier, verdict = benchmark_tier(category, pct)
        severity = 2 if tier in (0, 3) else 0
        candidates.append((severity, f"{label}: {pct}% ({num}/{int(den)} attempted) - {verdict}"))
    for label, exp_key, actual_key in EXPECTATION_PAIRS:
        exp, actual = try_float(extras.get(exp_key)), try_float(extras.get(actual_key))
        if exp is None or actual is None:
            continue
        diff = actual - exp
        perf = "overperforming" if diff > 0.5 else "underperforming" if diff < -0.5 else "in line with expectation"
        severity = 2 if perf != "in line with expectation" else 0
        candidates.append((severity, f"{label}: {actual} actual vs {exp} expected ({perf})"))
    if not candidates:
        return "No key match stats to present"
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def build_prompt(r, cat_avgs, category, club_name, standout_attr, standout_context):
    core = (
        f"{r['position']}, age {r['age']}, {r['appearances'] or 0} apps "
        f"({r['sub_appearances'] or 0} sub), {r['goals'] or 0} goals, "
        f"{r['assists'] or 0} assists, rating {r['avg_rating']}, "
        f"status {r['squad_status']}, CA {r['current_ability']}/PA {r['potential_ability']}, "
        f"contract {r['contract_expiry']}"
    )
    extras = r["attributes"] or {}
    relevant_keys = set(POSITION_KEY_ATTRIBUTES[category])

    fm_parts = []
    for dk, en in ALL_ATTRIBUTES:
        if dk not in relevant_keys:
            continue
        val = extras.get(dk)
        if val in (None, ""):
            continue
        avg = cat_avgs.get(dk)
        fm_parts.append(f"{en}={val} ({category} avg {avg})" if avg is not None and try_float(val) is not None else f"{en}={val}")

    context_parts = [f"{en}={extras.get(dk)}" for dk, en in MATCH_CONTEXT if extras.get(dk) not in (None, "")]
    all_stats = format_all_paired_stats(extras)

    return f"""Club: {club_name}
Player data: {core}
Position group: {category}
Position-relevant attributes only (1-20 scale): {', '.join(fm_parts)}
Match context: {', '.join(context_parts)}
Performance stats (min. {MIN_SAMPLE_SIZE} attempts, benchmarked against real professional football standards): {'; '.join(all_stats) if all_stats else 'none meet the minimum sample size'}

Assessment categories: {', '.join(ASSESSMENTS)}

Pre-selected standout data (computed, not your choice - use these directly):
- Standout position-relevant attribute (largest deviation from {category} average): {standout_attr}
- Standout performance stat (most extreme benchmark result): {standout_context}

Important: A player with low appearances is not automatically a development
case. Check age vs. CA/PA gap, consecutive matches not played, and whether
the club has marked them ready for release before concluding they deserve
more playing time.

Respond with ONLY this JSON object, nothing else:
{{
  "assessment": "<one of: {', '.join(ASSESSMENTS)}>",
  "reason": "<max 30 words. Must explain the assessment using the two standout data points given above, in your own words.>"
}}"""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/rotation-candidates")
def rotation_candidates(limit: int = 8):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("SELECT export_id FROM players ORDER BY exported_at DESC LIMIT 1")
    latest = cur.fetchone()
    if not latest:
        return {"error": "No player data found. Have you run an export yet?"}
    export_id = latest["export_id"]

    avgs_by_position = squad_averages_by_position(conn, export_id)

    cur.execute("""
        SELECT name, position, age, nation, squad_status, appearances,
               sub_appearances, goals, assists, avg_rating,
               current_ability, potential_ability, wage_monthly,
               contract_expiry, attributes
        FROM players
        WHERE export_id = %s
        ORDER BY COALESCE(appearances, 0) ASC
        LIMIT %s
    """, (export_id, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    club_name = "your club"
    for r in rows:
        club = (r["attributes"] or {}).get("Klub")
        if club:
            club_name = club
            break

    results = []
    for r in rows:
        extras = r["attributes"] or {}
        category = classify_position(r["position"])
        cat_avgs = avgs_by_position[category]

        standout_attr = pick_standout_attribute(extras, cat_avgs, category)
        standout_context = pick_standout_context(extras)

        prompt = build_prompt(r, cat_avgs, category, club_name, standout_attr, standout_context)
        raw = call_ollama(prompt)
        try:
            parsed = json.loads(raw)
            assessment = str(parsed.get("assessment", "")).strip()
            reason = str(parsed.get("reason", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            assessment, reason = "", raw

        if assessment not in ASSESSMENTS:
            assessment = f"[ukendt kategori: {assessment}]" if assessment else "[uventet svar - se raw]"
            reason = raw if not reason else reason

        results.append({
            "name": r["name"],
            "age": r["age"],
            "nation": r["nation"],
            "position": r["position"],
            "CA": r["current_ability"],
            "PA": r["potential_ability"],
            "Agreed squad role": extras.get("Spilletid"),
            "Actual squad role": extras.get("Faktisk spilletid"),
            "assessment": assessment,
            "Key attribute": standout_attr,
            "Key match stat": standout_context,
            "reason": reason,
        })

    unload_model()

    return {"players_considered": rows, "advice": results}
