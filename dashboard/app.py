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
import re
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
    for fname in ("summary.json", "generator_problem.json"):
        f = cycle_dir / fname
        if f.exists():
            result[fname.replace(".json", "")] = json.loads(f.read_text(encoding="utf-8"))
    # per-solver files: solver_<slug>.json, execution_<slug>.json, judgments_<slug>.json
    solvers_data: dict = {}
    for f in sorted(cycle_dir.glob("solver_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        solver_id = d.get("model", f.stem)
        solvers_data.setdefault(solver_id, {})["solution"] = d
    for f in sorted(cycle_dir.glob("execution_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        # match back by checking each solver slug
        for sid, sdata in solvers_data.items():
            slug = re.sub(r"[^\w-]", "_", sid)[:60]
            if f.name == f"execution_{slug}.json":
                sdata["execution"] = d
                break
    for f in sorted(cycle_dir.glob("judgments_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        solver_id = d.get("solver", "")
        if solver_id in solvers_data:
            solvers_data[solver_id]["judgments"] = d
    result["solvers_data"] = solvers_data
    return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def short_name(model_id: str) -> str:
    """Extract a readable short name from a model ID."""
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
            avg_scores = b.get("average_scores", {})
            best = max(avg_scores.items(), key=lambda x: x[1]) if avg_scores else ("", 0)
            rows.append({
                "Cycle":        b.get("cycle_num", ""),
                "Time (UTC)":   fmt_ts(b["timestamp"]) if b.get("timestamp") else "",
                "Generator":    short_name(b.get("generator", "")),
                "# Solvers":    len(b.get("solvers", [])),
                "# Judges":     len(b.get("judges_pool", [])),
                "Problem":      b.get("problem_title", ""),
                "Overall Avg":  round(b.get("overall_average", 0), 2),
                "Best Solver":  short_name(best[0]),
                "Best Score":   round(best[1], 2),
            })

        df_log = pd.DataFrame(rows)
        st.dataframe(df_log, use_container_width=True, hide_index=True)

        scores = [b.get("overall_average", 0) for b in battles]
        fig3 = px.histogram(
            x=scores,
            nbins=20,
            title="Distribution of Overall Average Scores",
            labels={"x": "Overall Average Score (0–10)", "y": "Count"},
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

        summary      = data.get("summary", {})
        problem      = data.get("generator_problem", {})
        solvers_data = data.get("solvers_data", {})

        # ── Header metrics
        avg_scores_map = summary.get("average_scores", {})
        best_pair = max(avg_scores_map.items(), key=lambda x: x[1]) if avg_scores_map else ("", 0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cycle",           summary.get("cycle_num", "—"))
        c2.metric("Overall Avg",     f"{summary.get('overall_average', 0):.2f} / 10")
        c3.metric("Best Solver",     short_name(best_pair[0]))
        c4.metric("Best Score",      f"{best_pair[1]:.2f} / 10")

        st.markdown(f"""
| Role | Models |
|---|---|
| **Generator** | `{summary.get('generator', '—')}` |
| **Solvers** | {len(summary.get('solvers', []))} models (all) |
| **Judges pool** | {len(summary.get('judges_pool', []))} models (all except generator) |
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

        # ── Scores summary table (all solvers ranked)
        with st.expander("🏆 Solver Rankings This Cycle", expanded=True):
            if avg_scores_map:
                rank_rows = sorted(
                    [{"Solver": short_name(k), "Avg Score": round(v, 2)} for k, v in avg_scores_map.items()],
                    key=lambda r: -r["Avg Score"],
                )
                for i, r in enumerate(rank_rows):
                    r["Rank"] = i + 1
                st.dataframe(
                    pd.DataFrame(rank_rows)[["Rank", "Solver", "Avg Score"]],
                    use_container_width=True, hide_index=True,
                )
                fig_bar = px.bar(
                    pd.DataFrame(rank_rows),
                    x="Avg Score", y="Solver", orientation="h",
                    title="Solver Scores This Cycle",
                    color="Avg Score", color_continuous_scale="RdYlGn",
                    range_color=[0, 10],
                )
                fig_bar.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                    height=max(300, len(rank_rows) * 40),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # ── Per-solver detail (selectbox)
        st.subheader("🔍 Per-Solver Detail")
        solver_ids = list(solvers_data.keys())
        if solver_ids:
            selected_solver = st.selectbox(
                "Select solver",
                solver_ids,
                format_func=lambda x: f"{short_name(x)}  ({avg_scores_map.get(x, 0):.2f}/10)",
            )
            sd = solvers_data.get(selected_solver, {})
            solution  = sd.get("solution", {})
            execution = sd.get("execution", {})
            judgments = sd.get("judgments", {})
            judge_results = judgments.get("judge_results", [])

            # Solution code
            with st.expander("💻 Solution Code", expanded=True):
                code = solution.get("solution_code", "")
                if code:
                    st.code(code, language="python")
                expl = solution.get("explanation", "")
                if expl:
                    st.markdown(f"**Explanation:** {expl}")

            # Execution
            with st.expander("🧪 Sandbox Execution", expanded=True):
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
                        use_container_width=True, hide_index=True,
                    )
                if execution.get("stderr"):
                    st.text_area("stderr", execution["stderr"], height=80)

            # Judge scores for this solver
            with st.expander("⚖️ Judge Scores", expanded=True):
                if judge_results:
                    score_rows = []
                    for jr in judge_results:
                        scores = jr.get("scores", {})
                        score_rows.append({
                            "Judge":       short_name(jr.get("judge", "")),
                            "Correctness": scores.get("correctness", ""),
                            "Efficiency":  scores.get("efficiency", ""),
                            "Readability": scores.get("readability", ""),
                            "Robustness":  scores.get("robustness", ""),
                            "Explanation": scores.get("explanation", ""),
                            "Overall":     jr.get("overall_score", ""),
                        })
                    st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

                    dims = ["correctness", "efficiency", "readability", "robustness", "explanation"]
                    avg_dim = []
                    for dim in dims:
                        vals = [jr.get("scores", {}).get(dim, 0) for jr in judge_results if jr.get("scores")]
                        avg_dim.append(sum(vals) / len(vals) if vals else 0)
                    fig_radar = go.Figure(go.Scatterpolar(
                        r=avg_dim + [avg_dim[0]],
                        theta=[d.capitalize() for d in dims] + [dims[0].capitalize()],
                        fill="toself",
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                        title="Score by Dimension",
                        height=360,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                    st.markdown("**Feedback:**")
                    for jr in judge_results:
                        fb = jr.get("feedback", "")
                        if fb:
                            st.markdown(f"> **{short_name(jr.get('judge', ''))}:** {fb}")

        # ── Elo snapshot
        with st.expander("🏆 Elo Snapshot After This Cycle"):
            elo_snap = summary.get("elo_after", {})
            if elo_snap:
                df_elo = pd.DataFrame(
                    [{"Model": short_name(k), "Elo": round(v, 2)}
                     for k, v in sorted(elo_snap.items(), key=lambda x: -x[1])]
                )
                df_elo.index += 1
                st.dataframe(df_elo, use_container_width=True)

