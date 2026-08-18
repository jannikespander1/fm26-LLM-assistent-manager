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
    "Football Manager 26. A recommendation has already been computed for "
    "you deterministically - your only job is to explain WHY it fits this "
    "specific player, grounded in the exact data given. CRITICAL RULE: "
    "only ever reference the exact attribute names, stat names, and values "
    "explicitly given to you. Never mention an attribute, stat, or metric "
    "that was not provided - not even a plausible-sounding one."
)

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
    ("Tilfredshed", "Happiness"),
    ("Tilfredshed med spilletid", "Playing Time Satisfaction"),
    ("Missede kampe i træk", "Consecutive matches NOT played"),
    ("Klar til at blive fritstillet", "Club marked as ready to release"),
]

PLAYER_BACKGROUND = [
    ("Personlighed", "Personality"),
    ("Spillertræk", "Traits"),
    ("Lejestatus", "Loan status"),
    ("Højde", "Height"),
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

# Delte tier-labels på tværs af ALLE kategorier - samme resultat vises altid ens
TIER_LABELS = ["poor", "below average", "good", "excellent"]

BENCHMARK_THRESHOLDS = {
    "pass": [65, 75, 85],
    "shot_conversion": [8, 12, 18],
    "shot_accuracy": [30, 40, 50],
    "tackle": [50, 60, 70],
    "aerial": [45, 55, 65],
}

TIER_TO_STAT_MAGNITUDE = {0: 3.0, 1: 1.0, 2: 1.0, 3: 3.0}

MIN_SAMPLE_SIZE = 5
MIN_MINUTES_FOR_EXPECTATION = 90

EXCEPTIONAL_THRESHOLD = 6.0
POSITIVE_THRESHOLD = 2.0
WEAK_THRESHOLD = 0.5

RECOMMENDATIONS = {
    ("Youth", "exceptional"): "Outstanding talent for his age - secure him a fixed place in the development plan, possibly ready for the first team sooner than usual.",
    ("Youth", "positive"): "Strong talent for his age - prioritize development playing time.",
    ("Youth", "weak_positive"): "Promising trend, but not yet conclusive - give a bit more playing time to confirm the direction.",
    ("Youth", "mixed"): "Mixed profile, natural for his age - monitor without urgent action.",
    ("Youth", "weak_negative"): "Weaknesses slightly outweigh strengths - worth following, not yet a concern.",
    ("Youth", "negative"): "Weaknesses still outweigh strengths - monitor development closely, avoid forcing playing time.",
    ("Youth", "severe_negative"): "Significantly below standard even for his age - reconsider whether he belongs in the academy long-term.",
    ("Youth", "insufficient"): "Not enough data for a reliable assessment - needs more playing time or scouting.",

    ("Early prime", "exceptional"): "Exceptional level for his age - future key player, give him responsibility now.",
    ("Early prime", "positive"): "Solid and improving level - give more playing time, looks ready for a bigger role.",
    ("Early prime", "weak_positive"): "Positive trend, but not conclusive - worth investing more playing time in.",
    ("Early prime", "mixed"): "Reliable depth without a clear strength yet - no urgent action needed.",
    ("Early prime", "weak_negative"): "Weaknesses slightly outweigh strengths - monitor, no need to rush.",
    ("Early prime", "negative"): "Doesn't match the squad's level right now - consider a loan for playing time elsewhere.",
    ("Early prime", "severe_negative"): "Significantly below standard at this stage - loan or sale should be seriously considered.",
    ("Early prime", "insufficient"): "Not enough data for a reliable assessment - continue monitoring development.",

    ("Prime age", "exceptional"): "Outstanding level in his prime - one of the squad's most important players, should be protected.",
    ("Prime age", "positive"): "Strong player in his prime - should be prioritized higher or retained.",
    ("Prime age", "weak_positive"): "Solid, with signs of more - may deserve a bigger role, not a clear upgrade yet.",
    ("Prime age", "mixed"): "Reliable, but not standout - fair rotation player.",
    ("Prime age", "weak_negative"): "Slightly below standard in his prime - worth watching, not urgent.",
    ("Prime age", "negative"): "In his prime, but not matching the required level - consider a loan/sale, limited time left to improve.",
    ("Prime age", "severe_negative"): "Significantly below standard mid-career - likely not part of the future.",
    ("Prime age", "insufficient"): "Not enough data for a reliable assessment - should be clarified quickly.",

    ("Veteran", "exceptional"): "Extraordinary level for his age - still a genuine key player, value him.",
    ("Veteran", "positive"): "Still performing above standard - valuable rotation/mentor player.",
    ("Veteran", "weak_positive"): "Still holding up fine, without being standout - fits well as experienced depth.",
    ("Veteran", "mixed"): "Mixed profile - use tactically as rotation, not a guaranteed starter.",
    ("Veteran", "weak_negative"): "Early signs of declining level - keep an eye on it, plan a successor.",
    ("Veteran", "negative"): "Past his best with no development potential left - consider contract termination, sale, or loan.",
    ("Veteran", "severe_negative"): "Significantly below standard with no way back - contract termination or exit should be prioritized.",
    ("Veteran", "insufficient"): "Not enough data for a reliable assessment - worth clarifying quickly given his age.",
}


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
    """Bruger nu delte TIER_LABELS - samme tier giver altid samme tekst, uanset kategori."""
    if pct is None:
        return None, None
    thresholds = BENCHMARK_THRESHOLDS[category]
    for i, upper in enumerate(thresholds):
        if pct < upper:
            return i, TIER_LABELS[i]
    return 3, TIER_LABELS[3]


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


def enrich_extras(r):
    extras = dict(r["attributes"] or {})
    extras["Mål"] = r["goals"]
    extras["Assists"] = r["assists"]
    return extras


def get_age_band(age):
    if age is None:
        return "Unknown"
    if age < 20:
        return "Youth"
    if age <= 23:
        return "Early prime"
    if age <= 29:
        return "Prime age"
    return "Veteran"


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


def pick_attribute(extras, cat_avgs, category, direction):
    relevant_keys = set(POSITION_KEY_ATTRIBUTES[category])
    best = None
    for dk, en in ALL_ATTRIBUTES:
        if dk not in relevant_keys:
            continue
        val = try_float(extras.get(dk))
        avg = cat_avgs.get(dk)
        if val is None or avg is None:
            continue
        dev = val - avg
        if direction == "positive" and dev <= 0:
            continue
        if direction == "negative" and dev >= 0:
            continue
        magnitude = abs(dev)
        if best is None or magnitude > best[0]:
            best = (magnitude, en, val, avg, dev)
    if not best:
        return f"No notable {direction} attribute in position-relevant set", 0.0
    magnitude, en, val, avg, dev = best
    word = "above" if dev > 0 else "below"
    return f"{en}={val} ({magnitude:.1f} {word} {category} avg {avg})", magnitude


def pick_context_stat(extras, direction):
    candidates = []
    for label, pct_key, num_key, den_key, category in PAIRED_STATS:
        pct = parse_percent(extras.get(pct_key))
        num, den = extras.get(num_key), try_float(extras.get(den_key))
        if pct is None or not den or den < MIN_SAMPLE_SIZE:
            continue
        tier, verdict = benchmark_tier(category, pct)
        if direction == "positive" and tier not in (2, 3):
            continue
        if direction == "negative" and tier not in (0, 1):
            continue
        weight = TIER_TO_STAT_MAGNITUDE[tier]
        candidates.append((weight, f"{label}: {pct}% ({num}/{int(den)} attempted) - {verdict}"))

    minutes = try_float(extras.get("Minutter"))
    if minutes is not None and minutes >= MIN_MINUTES_FOR_EXPECTATION:
        for label, exp_key, actual_key in EXPECTATION_PAIRS:
            exp, actual = try_float(extras.get(exp_key)), try_float(extras.get(actual_key))
            if exp is None or actual is None:
                continue
            diff = actual - exp
            if direction == "positive" and diff <= 0.5:
                continue
            if direction == "negative" and diff >= -0.5:
                continue
            perf = "overperforming" if diff > 0.5 else "underperforming"
            candidates.append((abs(diff), f"{label}: {actual} actual vs {exp} expected ({perf})"))

    if not candidates:
        return f"No notable {direction} match stat to present", 0.0
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1], candidates[0][0]


def compute_combined_signal(attr_pos, attr_neg, stat_pos, stat_neg):
    attribute_net = attr_pos - attr_neg
    stat_net = stat_pos - stat_neg
    combined_net = attribute_net + stat_net

    if attr_pos == 0 and attr_neg == 0 and stat_pos == 0 and stat_neg == 0:
        state = "insufficient"
    elif combined_net > EXCEPTIONAL_THRESHOLD:
        state = "exceptional"
    elif combined_net > POSITIVE_THRESHOLD:
        state = "positive"
    elif combined_net > WEAK_THRESHOLD:
        state = "weak_positive"
    elif combined_net >= -WEAK_THRESHOLD:
        state = "mixed"
    elif combined_net >= -POSITIVE_THRESHOLD:
        state = "weak_negative"
    elif combined_net >= -EXCEPTIONAL_THRESHOLD:
        state = "negative"
    else:
        state = "severe_negative"

    return state, attribute_net, stat_net


def build_prompt(r, extras, club_name, pos_attr, neg_attr, pos_stat, neg_stat, age_band, tier, attribute_net, stat_net, recommendation):
    background_parts = [f"{en}={extras.get(dk)}" for dk, en in PLAYER_BACKGROUND if extras.get(dk) not in (None, "")]

    core = (
        f"{r['position']}, age {r['age']}, {r['appearances'] or 0} apps "
        f"({r['sub_appearances'] or 0} sub, {extras.get('Minutter')} minutes), "
        f"{r['goals'] or 0} goals, {r['assists'] or 0} assists, rating {r['avg_rating']}, "
        f"status {r['squad_status']}, contract {r['contract_expiry']}, "
        f"club {club_name}, {', '.join(background_parts)}"
    )

    context_parts = [f"{en}={extras.get(dk)}" for dk, en in MATCH_CONTEXT if extras.get(dk) not in (None, "")]

    divergence_note = ""
    if (attribute_net > 0.5 and stat_net < -0.5) or (attribute_net < -0.5 and stat_net > 0.5):
        divergence_note = (
            "\n\nNOTE: The attribute signal and match stat signal point in DIFFERENT "
            "directions for this player. You MUST explicitly call this out in your "
            "reason - e.g. 'young player with strong underlying ability, but "
            "underperforming on [the specific stat given]', or the reverse."
        )

    return f"""Player data: {core}
Match context: {', '.join(context_parts)}

Age band: {age_band}
Signal tier: {tier}
Positive attribute (strongest, position-relevant): {pos_attr}
Negative attribute (weakest, position-relevant): {neg_attr}
Positive match stat: {pos_stat}
Negative match stat: {neg_stat}
{divergence_note}

RECOMMENDATION (already decided, computed - do not choose a different one): "{recommendation}"

Respond with ONLY this JSON object, nothing else:
{{
  "reason": "<max 50 words. Explain WHY the recommendation above fits this specific player. Reference the exact positive and negative data points given. If attribute and match stat signals diverge (see note above if present), explicitly say so.>"
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
        extras = enrich_extras(r)
        category = classify_position(r["position"])
        cat_avgs = avgs_by_position[category]

        pos_attr, attr_pos_mag = pick_attribute(extras, cat_avgs, category, "positive")
        neg_attr, attr_neg_mag = pick_attribute(extras, cat_avgs, category, "negative")
        pos_stat, stat_pos_mag = pick_context_stat(extras, "positive")
        neg_stat, stat_neg_mag = pick_context_stat(extras, "negative")
        age_band = get_age_band(r["age"])

        state, attribute_net, stat_net = compute_combined_signal(
            attr_pos_mag, attr_neg_mag, stat_pos_mag, stat_neg_mag
        )
        recommendation = RECOMMENDATIONS[(age_band, state)]

        prompt = build_prompt(r, extras, club_name, pos_attr, neg_attr, pos_stat, neg_stat,
                               age_band, state, attribute_net, stat_net, recommendation)
        raw = call_ollama(prompt)
        try:
            parsed = json.loads(raw)
            reason = str(parsed.get("reason", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            reason = raw

        results.append({
            "name": r["name"],
            "age": r["age"],
            "age_band": age_band,
            "minutes_played": extras.get("Minutter"),
            "club": extras.get("Klub"),
            "nation": r["nation"],
            "position": r["position"],
            "Agreed squad role": extras.get("Spilletid"),
            "Actual squad role": extras.get("Faktisk spilletid"),
            "tier": state,
            "recommendation": recommendation,
            "Positive key attribute": pos_attr,
            "Negative key attribute": neg_attr,
            "Positive key match stat": pos_stat,
            "Negative key match stat": neg_stat,
            "reason": reason,
        })

    unload_model()

    return {"players_considered": rows, "advice": results}
