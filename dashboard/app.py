"""
MatrixArena Dashboard — Placeholder (future Streamlit/Gradio UI).

This module will visualise the live leaderboard, Elo rating history,
and individual battle transcripts from data/leaderboard.json and
data/battles.jsonl.

To implement:
    pip install streamlit
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os


def load_leaderboard(path: str = "data/leaderboard.json") -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    print("MatrixArena Dashboard — UI not yet implemented.")
    print("Leaderboard snapshot:")
    for entry in load_leaderboard():
        print(f"  {entry['rank']}. {entry['model']}: {entry['elo']}")


if __name__ == "__main__":
    main()
