# src/load_data.py
import json
import pandas as pd
from pathlib import Path

def load_shots(events_path):
    shots = []

    for file in Path(events_path).glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            events = json.load(f)

        for ev in events:
            if ev.get("type", {}).get("name") == "Shot":
                shot = {}

                # Target
                outcome = ev.get("shot", {}).get("outcome", {}).get("name")
                shot["goal"] = 1 if outcome == "Goal" else 0

                # 👉 StatsBomb xG (SOLO per confronto, NON come feature)
                shot["statsbomb_xg"] = ev.get("shot", {}).get("statsbomb_xg")

                # Coordinate
                loc = ev.get("location", [None, None])
                shot["x"] = loc[0]
                shot["y"] = loc[1]

                # Categorical features
                shot["body_part"] = ev.get("shot", {}).get("body_part", {}).get("name", "Unknown")
                shot["shot_type"] = ev.get("shot", {}).get("type", {}).get("name", "Unknown")
                shot["technique"] = ev.get("shot", {}).get("technique", {}).get("name", "Unknown")
                shot["play_pattern"] = ev.get("play_pattern", {}).get("name", "Unknown")

                # Boolean
                shot["under_pressure"] = ev.get("under_pressure", False)

                # Time
                shot["minute"] = ev.get("minute", 0)
                shot["second"] = ev.get("second", 0)

                shots.append(shot)

    return pd.DataFrame(shots)
