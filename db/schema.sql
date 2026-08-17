CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    export_id UUID NOT NULL,
    exported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    name TEXT NOT NULL,
    position TEXT,
    age INT,
    nation TEXT,
    squad_status TEXT,
    wage_monthly NUMERIC,
    value_low NUMERIC,
    value_high NUMERIC,
    contract_expiry DATE,
    current_ability NUMERIC,
    potential_ability NUMERIC,
    appearances INT,
    sub_appearances INT,
    goals INT,
    assists INT,
    avg_rating NUMERIC,
    attributes JSONB
);

CREATE INDEX IF NOT EXISTS idx_players_export_id ON players(export_id);
CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);
