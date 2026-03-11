"""
MatrixArena Dashboard — Streamlit UI

Run with:
    streamlit run dashboard/app.py

Reads from (all git-ignored runtime outputs):
    data/leaderboard.json
    data/battles.jsonl
    data/cycles/<dir>/
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_DATA = _ROOT / "data"
_LEADERBOARD = _DATA / "leaderboard.json"
_BATTLES = _DATA / "battles.jsonl"
_CYCLES = _DATA / "cycles"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def load_leaderboard() -> list[dict]:
    if not _LEADERBOARD.exists():
        return []
    return json.loads(_LEADERBOARD.read_text(encoding="utf-8"))


@st.cache_data(ttl=10)
def load_battles() -> list[dict]:
    if not _BATTLES.exists():
        return []
    rows = []
    for line in _BATTLES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


@st.cache_data(ttl=10)
def load_cycle_dirs() -> list[Path]:
    if not _CYCLES.exists():
        return []
    dirs = sorted(
        [d for d in _CYCLES.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return dirs


def load_cycle(cycle_dir: Path) -> dict:
    result = {}
    for fname in (
        "summary.json",
        "generator_problem.json",
        "solver_solution.json",
        "execution_result.json",
    ):
        f = cycle_dir / fname
        if f.exists():
            result[fname.replace(".json", "")] = json.loads(f.read_text(encoding="utf-8"))
    judges = []
    for f in sorted(cycle_dir.glob("judge_*.json")):
        judges.append(json.loads(f.read_text(encoding="utf-8")))
    result["judges"] = judges
    return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def short_name(model_id: str) -> str:
    """Extract a readable short name from a model ID."""
    # openrouter/provider/model-name  →  model-name
    parts = model_id.split("/")
    return parts[-1] if parts else model_id


def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MatrixArena",
    page_icon="🏟️",
    layout="wide",
)

st.title("🏟️ MatrixArena — LLM Peer-Review Leaderboard")
st.caption("Auto-refreshes every 10 seconds · data sourced from `data/`")

if st.button("🔄 Refresh now"):
    st.cache_data.clear()
    st.rerun()

tab_lb, tab_hist, tab_log, tab_cycle = st.tabs(
    ["📊 Leaderboard", "📈 Elo History", "📋 Battle Log", "🔍 Cycle Detail"]
)


# ===========================================================================
# Tab 1 · Leaderboard
# ===========================================================================
with tab_lb:
    lb = load_leaderboard()

    if not lb:
        st.info("No leaderboard data yet. Run `python main.py --cycles 1` to generate results.")
    else:
        df_lb = pd.DataFrame(lb)
        df_lb["short_name"] = df_lb["model"].apply(short_name)
        df_lb = df_lb.sort_values("elo", ascending=False).reset_index(drop=True)
        df_lb.index += 1

        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            fig = px.bar(
                df_lb,
                x="elo",
                y="short_name",
                orientation="h",
                title="Current Elo Ratings",
                labels={"elo": "Elo Rating", "short_name": ""},
                color="elo",
                color_continuous_scale="RdYlGn",
                text="elo",
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
                height=max(350, len(df_lb) * 42),
                margin=dict(l=0, r=60, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.subheader("Rankings")
            st.dataframe(
                df_lb[["rank", "short_name", "elo"]].rename(
                    columns={"rank": "#", "short_name": "Model", "elo": "Elo"}
                ),
                use_container_width=True,
                hide_index=True,
            )


# ===========================================================================
# Tab 2 · Elo History
# ===========================================================================
with tab_hist:
    battles = load_battles()

    if not battles:
        st.info("No battle data yet.")
    else:
        # Build a tidy time-series from elo_after snapshots
        records = []
        for b in battles:
            elo_snap = b.get("elo_after", {})
            cycle = b.get("cycle_num", 0)
            ts = b.get("timestamp", 0)
            for model_id, rating in elo_snap.items():
                records.append(
                    {"cycle": cycle, "timestamp": ts, "model": short_name(model_id), "elo": rating}
                )

        df_hist = pd.DataFrame(records)
        all_models = sorted(df_hist["model"].unique())

        selected = st.multiselect(
            "Filter models",
            options=all_models,
            default=all_models,
        )
        df_filtered = df_hist[df_hist["model"].isin(selected)]

        fig2 = px.line(
            df_filtered,
            x="cycle",
            y="elo",
            color="model",
            markers=True,
            title="Elo Rating History per Cycle",
            labels={"cycle": "Cycle", "elo": "Elo Rating", "model": "Model"},
        )
        fig2.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=-0.4))
        st.plotly_chart(fig2, use_container_width=True)


# ===========================================================================
# Tab 3 · Battle Log
# ===========================================================================
with tab_log:
    battles = load_battles()

    if not battles:
        st.info("No battle log yet.")
    else:
        rows = []
        for b in battles:
            rows.append({
                "Cycle":     b.get("cycle_num", ""),
                "Time (UTC)": fmt_ts(b["timestamp"]) if b.get("timestamp") else "",
                "Generator": short_name(b.get("generator", "")),
                "Solver":    short_name(b.get("solver", "")),
                "# Judges":  len(b.get("judges", [])),
                "Problem":   b.get("problem_title", ""),
                "Exec":      b.get("exec_status", ""),
                "✅ Pass":   b.get("exec_passed", ""),
                "❌ Fail":   b.get("exec_failed", ""),
                "Avg Score": round(b.get("average_score", 0), 2),
            })

        df_log = pd.DataFrame(rows)
        st.dataframe(df_log, use_container_width=True, hide_index=True)

        # Score distribution
        scores = [b.get("average_score", 0) for b in battles]
        fig3 = px.histogram(
            x=scores,
            nbins=20,
            title="Distribution of Average Scores",
            labels={"x": "Average Judge Score (0–10)", "y": "Count"},
        )
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)


# ===========================================================================
# Tab 4 · Cycle Detail
# ===========================================================================
with tab_cycle:
    cycle_dirs = load_cycle_dirs()

    if not cycle_dirs:
        st.info("No cycle artifacts yet. Run `python main.py` to generate results.")
    else:
        dir_options = {d.name: d for d in cycle_dirs}
        selected_dir = st.selectbox("Select cycle", list(dir_options.keys()))
        data = load_cycle(dir_options[selected_dir])

        summary = data.get("summary", {})
        problem = data.get("generator_problem", {})
        solution = data.get("solver_solution", {})
        execution = data.get("execution_result", {})
        judges = data.get("judges", [])

        # ── Header metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Cycle",     summary.get("cycle_num", "—"))
        c2.metric("Avg Score", f"{summary.get('average_score', 0):.2f} / 10")
        c3.metric("Exec",      summary.get("exec_status", "—"))
        c4.metric("Passed",    summary.get("exec_passed", "—"))
        c5.metric("Failed",    summary.get("exec_failed", "—"))

        st.markdown(f"""
| Role | Model |
|---|---|
| **Generator** | `{summary.get('generator', '—')}` |
| **Solver** | `{summary.get('solver', '—')}` |
| **Judges** | {', '.join(f'`{j}`' for j in summary.get('judges', []))} |
""")

        # ── Generator Problem
        with st.expander("📝 Generator — Problem", expanded=True):
            st.markdown(f"### {problem.get('title', 'Untitled')}")
            st.markdown(problem.get("description", ""))
            tc = problem.get("test_cases", [])
            if tc:
                st.markdown("**Test Cases:**")
                st.dataframe(pd.DataFrame(tc), use_container_width=True, hide_index=True)
            crit = problem.get("evaluation_criteria", "")
            if crit:
                st.markdown(f"**Evaluation Criteria:** {crit}")

        # ── Solver Solution
        with st.expander("💻 Solver — Solution", expanded=True):
            code = solution.get("solution_code", "")
            if code:
                st.code(code, language="python")
            expl = solution.get("explanation", "")
            if expl:
                st.markdown(f"**Explanation:** {expl}")

        # ── Execution Result
        with st.expander("🧪 Sandbox Execution Result", expanded=True):
            st.markdown(f"**Status:** `{execution.get('status', '—')}`  ·  {execution.get('summary', '')}")
            tests = execution.get("tests", [])
            if tests:
                df_tests = pd.DataFrame(tests)
                def _color(val):
                    colors = {"pass": "background-color:#1a4d2e", "fail": "background-color:#5c1a1a",
                              "error": "background-color:#4d3a00", "skip": ""}
                    return colors.get(val, "")
                st.dataframe(
                    df_tests.style.applymap(_color, subset=["status"]),
                    use_container_width=True,
                    hide_index=True,
                )
            if execution.get("stderr"):
                st.text_area("stderr", execution["stderr"], height=100)

        # ── Judge Scores
        with st.expander("⚖️ Judge Scores & Feedback", expanded=True):
            if judges:
                score_rows = []
                for jr in judges:
                    scores = jr.get("scores", {})
                    score_rows.append({
                        "Judge":        short_name(jr.get("model", jr.get("judge", ""))),
                        "Correctness":  scores.get("correctness", ""),
                        "Efficiency":   scores.get("efficiency", ""),
                        "Readability":  scores.get("readability", ""),
                        "Robustness":   scores.get("robustness", ""),
                        "Explanation":  scores.get("explanation", ""),
                        "Overall":      jr.get("overall_score", ""),
                    })
                st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

                # Radar chart (average across judges)
                dims = ["correctness", "efficiency", "readability", "robustness", "explanation"]
                avg_scores = []
                for dim in dims:
                    vals = [jr.get("scores", {}).get(dim, 0) for jr in judges if jr.get("scores")]
                    avg_scores.append(sum(vals) / len(vals) if vals else 0)

                fig_radar = go.Figure(go.Scatterpolar(
                    r=avg_scores + [avg_scores[0]],
                    theta=[d.capitalize() for d in dims] + [dims[0].capitalize()],
                    fill="toself",
                    name="Avg Judge Score",
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                    title="Average Score by Dimension",
                    height=380,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                st.markdown("**Judge Feedback:**")
                for jr in judges:
                    feedback = jr.get("feedback", "")
                    if feedback:
                        model = short_name(jr.get("model", jr.get("judge", "")))
                        st.markdown(f"> **{model}:** {feedback}")

        # ── Elo after this cycle
        with st.expander("🏆 Elo Snapshot After This Cycle"):
            elo_snap = summary.get("elo_after", {})
            if elo_snap:
                df_elo = pd.DataFrame(
                    [{"Model": short_name(k), "Elo": round(v, 2)} for k, v in sorted(elo_snap.items(), key=lambda x: -x[1])]
                )
                df_elo.index += 1
                st.dataframe(df_elo, use_container_width=True)

