import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

EXPORT_DIR = Path("/mnt/c/Users/janni/Documents/Sports Interactive/Football Manager 26/FM26PlayerExport by vinteset/Exports CSV")

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "fm26",
    "user": "fm26",
    "password": os.environ["POSTGRES_PASSWORD"],
}

# --- Parsing: rå CSV-tekst -> rigtige typer -----------------------------

def parse_decimal(val):
    """'7,05' -> 7.05 (dansk komma til punktum)"""
    if pd.isna(val) or str(val).strip() in ("-", ""):
        return None
    try:
        return float(str(val).strip().replace(",", "."))
    except ValueError:
        return None

def parse_money(val):
    """'3mio. kr/måned' -> 3000000.0, '80K kr/måned' -> 80000.0"""
    if pd.isna(val) or str(val).strip() in ("-", ""):
        return None
    m = re.match(r"([\d,]+)\s*(mio\.|K)?", str(val).strip())
    if not m:
        return None
    num = float(m.group(1).replace(",", "."))
    if m.group(2) == "mio.":
        return num * 1_000_000
    if m.group(2) == "K":
        return num * 1_000
    return num

def parse_value_range(val):
    """'44mio. kr - 52mio. kr' -> (44000000.0, 52000000.0)"""
    if pd.isna(val) or str(val).strip() in ("-", ""):
        return None, None
    parts = str(val).split(" - ")
    if len(parts) == 2:
        return parse_money(parts[0]), parse_money(parts[1])
    v = parse_money(parts[0])
    return v, v

def parse_appearances(val):
    """'31 (6)' -> (31, 6) = 31 kampe, 6 som indskifter"""
    if pd.isna(val) or str(val).strip() in ("-", ""):
        return None, None
    m = re.match(r"(\d+)(?:\s*\((\d+)\))?", str(val).strip())
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2)) if m.group(2) else 0

def parse_danish_date(val):
    """'30/6/2034' -> date(2034, 6, 30)"""
    if pd.isna(val) or str(val).strip() in ("-", ""):
        return None
    try:
        return datetime.strptime(str(val).strip(), "%d/%m/%Y").date()
    except ValueError:
        return None

def last_column_named(df, name):
    """'Udløber' står to gange i eksporten - pandas omdøber nr. 2 til
    'Udløber.1'. Vi vil have den SIDSTE (den ved siden af Løn/Evne =
    kontraktudløb), ikke den midt i Frikøbsklausul-blokken."""
    matches = [c for c in df.columns if c == name or c.startswith(name + ".")]
    return matches[-1] if matches else None

MAPPED_COLUMNS = {"Spiller", "Position", "Alder", "Nation", "Hierarki",
                   "Løn", "Transferværdi", "Evne", "Potentiale",
                   "Optrædener", "Mål", "Assists", "Bedømmelse"}

def insert_players(df):
    export_id = str(uuid.uuid4())
    exported_at = datetime.now()
    udloeber_col = last_column_named(df, "Udløber")
    mapped = MAPPED_COLUMNS | ({udloeber_col} if udloeber_col else set())

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    for _, row in df.iterrows():
        value_low, value_high = parse_value_range(row.get("Transferværdi"))
        appearances, sub_appearances = parse_appearances(row.get("Optrædener"))

        # Alt der ikke er eksplicit mappet, ender i attributes-JSONB'en
        extras = {c: (None if pd.isna(row[c]) else row[c]) for c in df.columns if c not in mapped}

        cur.execute(
            """
            INSERT INTO players (
                export_id, exported_at, name, position, age, nation,
                squad_status, wage_monthly, value_low, value_high,
                contract_expiry, current_ability, potential_ability,
                appearances, sub_appearances, goals, assists, avg_rating,
                attributes
            ) VALUES (
                %(export_id)s, %(exported_at)s, %(name)s, %(position)s, %(age)s, %(nation)s,
                %(squad_status)s, %(wage_monthly)s, %(value_low)s, %(value_high)s,
                %(contract_expiry)s, %(current_ability)s, %(potential_ability)s,
                %(appearances)s, %(sub_appearances)s, %(goals)s, %(assists)s, %(avg_rating)s,
                %(attributes)s
            )
            """,
            {
                "export_id": export_id,
                "exported_at": exported_at,
                "name": row.get("Spiller"),
                "position": row.get("Position"),
                "age": int(row["Alder"]) if pd.notna(row.get("Alder")) else None,
                "nation": row.get("Nation"),
                "squad_status": row.get("Hierarki"),
                "wage_monthly": parse_money(row.get("Løn")),
                "value_low": value_low,
                "value_high": value_high,
                "contract_expiry": parse_danish_date(row.get(udloeber_col)) if udloeber_col else None,
                "current_ability": parse_decimal(row.get("Evne")),
                "potential_ability": parse_decimal(row.get("Potentiale")),
                "appearances": appearances,
                "sub_appearances": sub_appearances,
                "goals": int(row["Mål"]) if pd.notna(row.get("Mål")) else None,
                "assists": int(row["Assists"]) if pd.notna(row.get("Assists")) else None,
                "avg_rating": parse_decimal(row.get("Bedømmelse")),
                "attributes": psycopg2.extras.Json(extras),
            },
        )

    conn.commit()
    cur.close()
    conn.close()
    return len(df), export_id

# --- Watcher -------------------------------------------------------------

def wait_until_stable(path, checks=10, interval=0.5):
    last_size = -1
    for _ in range(checks):
        if not path.exists():
            return False
        size = path.stat().st_size
        if size > 0 and size == last_size:
            return True
        last_size = size
        time.sleep(interval)
    return False

class ExportHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".csv":
            return

        print(f"Ny fil opdaget: {path.name}")
        if not wait_until_stable(path):
            print(f"Filen blev aldrig stabil: {path.name}")
            return

        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8")
            count, export_id = insert_players(df)
            print(f"Gemt {count} spillere i databasen (export_id={export_id})")
        except Exception as e:
            print(f"Fejl under indlæsning: {e}")

if __name__ == "__main__":
    observer = Observer()
    observer.schedule(ExportHandler(), str(EXPORT_DIR), recursive=False)
    observer.start()
    print(f"Overvåger: {EXPORT_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
