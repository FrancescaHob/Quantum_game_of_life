import os
import subprocess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from functools import lru_cache

import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table, callback_context
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "simulation_data"
INDEX_CSV = DATA_DIR / "index.csv"
VENV_PYTHON = BASE_DIR / ".venv/Scripts/python.exe"

# ---------- Load catalog ----------
if not INDEX_CSV.exists():
    raise FileNotFoundError(f"Missing {INDEX_CSV}. Run simulations first (index at: {INDEX_CSV}).")

index = pd.read_csv(INDEX_CSV)

# Ensure expected columns exist
for col in [
    "path", "entropy_csv", "grid_size", "p_dead", "meas_density", "meas_interval",
    "seed_amp", "seed_phase", "run_id", "generations", "final_entropy_bits", "tail_var_entropy"
]:
    if col not in index.columns:
        index[col] = pd.NA

# Categorical helps with dropdowns
for col in ["grid_size", "p_dead", "meas_density", "meas_interval", "seed_amp", "seed_phase"]:
    if col in index.columns and not index[col].isna().all():
        index[col] = index[col].astype("category")

# ---------- Small in-process cache of run data ----------
# Returns (gens: np.ndarray[int], N: int, p_live_3d: float[G,N,N], cumsum: float[G+1,N,N])
@lru_cache(maxsize=4)
def load_run_p_live(path: str):
    table = pq.read_table(path, columns=["generation","i","j","p_live","grid_size"], use_threads=True)
    df = table.to_pandas()

    gens = np.sort(df["generation"].unique().astype(int))
    N = int(df["grid_size"].iloc[0]) if "grid_size" in df.columns else int(np.sqrt((df["generation"] == gens[0]).sum()))
    G = len(gens)

    p_live_3d = np.zeros((G, N, N), dtype=np.float32)
    # faster vector fill
    for g_idx, g in enumerate(gens):
        snap = df[df["generation"] == g]
        ii = snap["i"].to_numpy(dtype=int, copy=False)
        jj = snap["j"].to_numpy(dtype=int, copy=False)
        vv = snap["p_live"].to_numpy(dtype=np.float32, copy=False)
        p_live_3d[g_idx, ii, jj] = vv

    # cumulative sum along generations for O(1) window averages
    cumsum = np.concatenate([np.zeros((1, N, N), dtype=np.float32), np.cumsum(p_live_3d, axis=0)], axis=0)
    return gens, N, p_live_3d, cumsum

def build_entropy_figure(entropy_csv_path, G, current_gidx, mode, win_size):
    fig = go.Figure()
    if entropy_csv_path and os.path.exists(entropy_csv_path):
        e = pd.read_csv(entropy_csv_path)
        fig.add_trace(go.Scatter(x=e["generation"], y=e["entropy_bits"],
                                 mode="lines", name="block entropy (bits)"))
        # vertical cursor
        if current_gidx is not None and 0 <= current_gidx < G:
            x = int(e["generation"].iloc[current_gidx])
            fig.add_vline(x=x, line_width=1, line_dash="dot", line_color="gray")

        # shaded window for "Windowed average"
        if mode == "window" and win_size and current_gidx is not None:
            half = max(1, int(win_size)) - 1
            g_start = int(max(e["generation"].iloc[0], e["generation"].iloc[current_gidx] - half + 1))
            g_end   = int(e["generation"].iloc[current_gidx])
            fig.add_vrect(x0=g_start, x1=g_end, fillcolor="LightSkyBlue", opacity=0.25, line_width=0)
        fig.update_layout(margin=dict(l=30,r=10,t=30,b=30), height=280)
    return fig

def build_heatmap(p_live_2d, title):
    fig_hm = px.imshow(p_live_2d, origin="upper", color_continuous_scale="viridis", title=title)
    fig_hm.update_layout(margin=dict(l=30,r=10,t=40,b=30), height=420)
    return fig_hm

# ---------- App ----------
app = Dash(__name__)
app.title = "Quantum GoL — Run Explorer"

# Controls
controls = html.Div([
    html.H2("Filters"),
    html.Div([
        html.Label("grid_size"),
        dcc.Dropdown(
            options=sorted([int(x) for x in index["grid_size"].dropna().unique()]) if "grid_size" in index else [],
            multi=True,
            id="f-gsize"
        ),
    ]),
    html.Div([
        html.Label("p_dead"),
        dcc.Dropdown(sorted(index["p_dead"].dropna().unique()), multi=True, id="f-pdead"),
    ]),
    html.Div([
        html.Label("meas_density"),
        dcc.Dropdown(sorted(index["meas_density"].dropna().unique()), multi=True, id="f-mdens"),
    ]),
    html.Div([
        html.Label("meas_interval"),
        dcc.Dropdown(sorted(index["meas_interval"].dropna().unique()), multi=True, id="f-mint"),
    ]),
    html.Div([
        html.Label("seed_amp"),
        dcc.Dropdown(sorted(index["seed_amp"].dropna().unique()), multi=True, id="f-samp"),
    ]),
    html.Div([
        html.Label("seed_phase"),
        dcc.Dropdown(sorted(index["seed_phase"].dropna().unique()), multi=True, id="f-sphase"),
    ]),
    html.Div([
        html.Label("min generations"),
        dcc.Input(id="f-min-gens", type="number", value=0, min=0, step=10, style={"width":"100%"}),
    ], style={"marginTop":"8px"}),
    html.Div([
        html.Label("max final entropy (bits)"),
        dcc.Input(id="f-max-ent", type="number", value=None, step=0.1, style={"width":"100%"}),
    ], style={"marginTop":"8px"}),
    # New filter: Minimum final entropy
    html.Div([
        html.Label("min final entropy (bits)"),
        dcc.Input(id="f-min-ent", type="number", value=None, step=0.1, style={"width":"100%"}),
    ], style={"marginTop":"8px"}),
], style={"width":"22%", "display":"inline-block", "verticalAlign":"top", "padding":"10px"})

# Table
table = html.Div([
    html.H2("Runs"),
    dash_table.DataTable(
        id="runs-table",
        columns=[
            {"name":"path", "id":"path"},
            {"name":"grid_size", "id":"grid_size"},
            {"name":"p_dead", "id":"p_dead"},
            {"name":"meas_density", "id":"meas_density"},
            {"name":"meas_interval", "id":"meas_interval"},
            {"name":"seed_amp", "id":"seed_amp"},
            {"name":"seed_phase", "id":"seed_phase"},
            {"name":"run_id", "id":"run_id"},
            {"name":"generations", "id":"generations"},
            {"name":"final_entropy_bits", "id":"final_entropy_bits"},
            {"name":"tail_var_entropy", "id":"tail_var_entropy"},
        ],
        data=index.to_dict("records"),
        row_selectable="single",
        page_size=12,
        sort_action="native",
        filter_action="none",
        style_table={"height":"520px","overflowY":"auto"},
        style_cell={"fontFamily":"monospace", "fontSize":13, "whiteSpace":"nowrap", "textOverflow":"ellipsis", "maxWidth":0},
    ),
], style={"width":"76%", "display":"inline-block", "verticalAlign":"top", "padding":"10px"})

# Right panel: run controls & graphs
run_controls = html.Div([
    html.H2("Selected run"),
    html.Div(id="run-meta", style={"marginBottom":"8px", "fontFamily":"monospace"}),

    html.Div([
        html.Label("View mode"),
        dcc.RadioItems(
            id="view-mode",
            options=[
                {"label":"Snapshot", "value":"snapshot"},
                {"label":"Cumulative", "value":"cumulative"},
                {"label":"Windowed average", "value":"window"},
                {"label":"Final cumulative", "value":"final"},
            ],
            value="snapshot",
            inline=True
        ),
    ], style={"marginBottom":"6px"}),

    html.Div([
        html.Label("Generation"),
        dcc.Slider(id="gen-slider", min=1, max=1, step=1, value=1, tooltip={"placement":"bottom","always_visible":True}),
    ], style={"marginBottom":"10px"}),

    html.Div([
        html.Label("Window size (gens)"),
        dcc.Slider(id="win-size", min=1, max=50, step=1, value=10, tooltip={"placement":"bottom","always_visible":True}),
    ], id="win-size-wrap", style={"display":"none", "marginBottom":"10px"}),

    dcc.Graph(id="entropy-graph"),
    html.Div([
        html.Button("◀ Prev run", id="btn-prev-run", n_clicks=0, style={"marginRight":"8px"}),
        html.Button("Next run ▶", id="btn-next-run", n_clicks=0, style={"marginRight":"16px"}),
        html.Button("Replay in Pygame", id="btn-replay", n_clicks=0),
        html.Span(id="replay-status", style={"marginLeft":"10px"})
    ], style={"marginTop":"6px"}),
    dcc.Graph(id="preview-heatmap"),
])

app.layout = html.Div([
    html.H1("Quantum GoL — Run Explorer"),
    html.Div([controls, table]),
    html.Hr(),
    run_controls
], style={"maxWidth":"1400px", "margin":"0 auto"})

# ---------- Helpers ----------
def _apply_filters(df, gsize, pdead, mdens, mint, samp, sphase, min_gens, max_ent, min_ent):
    """
    Applies all filters to the DataFrame.
    min_ent has been added to the signature.
    """
    filt = df.copy()
    if gsize:  filt = filt[filt["grid_size"].isin(gsize)]
    if pdead:  filt = filt[filt["p_dead"].isin(pdead)]
    if mdens:  filt = filt[filt["meas_density"].isin(mdens)]
    if mint:   filt = filt[filt["meas_interval"].isin(mint)]
    if samp:   filt = filt[filt["seed_amp"].isin(samp)]
    if sphase: filt = filt[filt["seed_phase"].isin(sphase)]
    if min_gens is not None:
        filt = filt[filt["generations"] >= int(min_gens)]
    if max_ent is not None and "final_entropy_bits" in filt.columns:
        filt = filt[filt["final_entropy_bits"] <= float(max_ent)]
    # Filter by minimum final entropy
    if min_ent is not None and "final_entropy_bits" in filt.columns:
        filt = filt[filt["final_entropy_bits"] >= float(min_ent)]
    return filt

# ---------- Callbacks ----------
@app.callback(
    Output("runs-table", "data"),
    Input("f-gsize", "value"),
    Input("f-pdead", "value"),
    Input("f-mdens", "value"),
    Input("f-mint", "value"),
    Input("f-samp", "value"),
    Input("f-sphase", "value"),
    Input("f-min-gens", "value"),
    Input("f-max-ent", "value"),
    Input("f-min-ent", "value"),  # Added new Input
)
def update_table(gsize, pdead, mdens, mint, samp, sphase, min_gens, max_ent, min_ent):  # Added new argument
    filt = _apply_filters(index, gsize, pdead, mdens, mint, samp, sphase, min_gens, max_ent, min_ent)  # Passed new argument
    return filt.to_dict("records")

@app.callback(
    Output("run-meta", "children"),
    Output("gen-slider", "min"),
    Output("gen-slider", "max"),
    Output("gen-slider", "value"),
    Input("runs-table", "selected_rows"),
    State("runs-table", "data"),
    State("gen-slider", "value"),
    prevent_initial_call=True
)
def on_select_run(selected, rows, current_gen_value):
    if not selected:
        return dash.no_update, 1, 1, 1

    row = rows[selected[0]]
    path = row["path"]
    gens, N, p_live_3d, cumsum = load_run_p_live(path)

    meta = (
        f"path={path} | N={N} | gens={len(gens)} | p_dead={row.get('p_dead')} | "
        f"D={row.get('meas_density')} | I={row.get('meas_interval')} | "
        f"A={row.get('seed_amp')} | P={row.get('seed_phase')}"
    )

    min_g = int(gens[0])
    max_g = int(gens[-1])

    # Preserve current generation if possible, otherwise clamp
    if current_gen_value is None:
        init_g = min_g
    else:
        # snap to nearest valid generation in this run
        # If generations are contiguous integers, simple clamp suffices:
        init_g = int(np.clip(int(current_gen_value), min_g, max_g))
        # If your gens are not guaranteed contiguous, map to nearest:
        # idx = int(np.argmin(np.abs(gens - int(current_gen_value))))
        # init_g = int(gens[idx])

    return meta, min_g, max_g, init_g



@app.callback(
    Output("entropy-graph", "figure"),
    Output("preview-heatmap", "figure"),
    Output("win-size-wrap", "style"),
    Input("view-mode", "value"),
    Input("gen-slider", "value"),
    Input("win-size", "value"),
    Input("runs-table", "selected_rows"),
    State("runs-table", "data"),
    prevent_initial_call=True
)
def update_graphs(view_mode, gen_value, win_size, selected, rows):
    if not selected or not rows:
        return go.Figure(), go.Figure(), {"display":"none"}
    row = rows[selected[0]]
    path = row["path"]
    entropy_csv = row.get("entropy_csv")
    gens, N, p_live_3d, cumsum = load_run_p_live(path)

    # map generation value to index
    # (slider uses real generation numbers; gens may start at 0 or 1)
    gen_value = int(gen_value)
    # find nearest index
    try:
        current_gidx = int(np.where(gens == gen_value)[0][0])
    except IndexError:
        current_gidx = int(np.argmin(np.abs(gens - gen_value)))
        gen_value = int(gens[current_gidx])

    # Heatmap according to mode
    if view_mode == "snapshot":
        frame = p_live_3d[current_gidx]
        title = f"Snapshot p_live (gen {gen_value})"
        show_win = {"display":"none"}

    elif view_mode == "cumulative":
        # average from start..current
        total = cumsum[current_gidx + 1]  # inclusive
        denom = (current_gidx + 1)
        frame = total / max(1, denom)
        title = f"Cumulative average p_live (0..{gen_value})"
        show_win = {"display":"none"}

    elif view_mode == "final":
        total = cumsum[-1]
        denom = p_live_3d.shape[0]
        frame = total / max(1, denom)
        title = f"Final cumulative average p_live (0..{int(gens[-1])})"
        show_win = {"display":"none"}

    else:  # "window"
        w = max(1, int(win_size))
        start = max(0, current_gidx - w + 1)
        end = current_gidx + 1
        total = cumsum[end] - cumsum[start]
        denom = (end - start)
        frame = total / max(1, denom)
        title = f"Windowed average p_live (gen {int(gens[start])}..{gen_value}, w={denom})"
        show_win = {"display": "block"}

    fig_ent = build_entropy_figure(entropy_csv, len(gens), current_gidx, view_mode, win_size)
    fig_hm  = build_heatmap(frame, title)
    return fig_ent, fig_hm, show_win

# Click on entropy graph to set generation slider
@app.callback(
    Output("gen-slider", "value", allow_duplicate=True),
    Input("entropy-graph", "clickData"),
    State("gen-slider", "value"),
    prevent_initial_call=True
)
def jump_to_clicked_generation(clickData, current_value):
    if not clickData or "points" not in clickData:
        return current_value
    x = clickData["points"][0].get("x")
    try:
        return int(x)
    except Exception:
        return current_value


@app.callback(
    Output("replay-status", "children"),
    Input("btn-replay", "n_clicks"),
    State("runs-table", "selected_rows"),
    State("runs-table", "data"),
    prevent_initial_call=True
)
def replay_run(n, selected, rows):
    if not selected:
        return "Select a run first."
    row = rows[selected[0]]
    path = row["path"]
    if not os.path.exists(path):
        return "Parquet not found."
    try:
        subprocess.Popen([str(VENV_PYTHON), str(BASE_DIR / "replay_run.py"), path])
        return "Launching replay..."
    except Exception as e:
        return f"Failed to launch: {e}"

# Navigation buttons    
@app.callback(
    Output("runs-table", "selected_rows"),
    Input("btn-prev-run", "n_clicks"),
    Input("btn-next-run", "n_clicks"),
    State("runs-table", "selected_rows"),
    State("runs-table", "data"),
    prevent_initial_call=True
)
def nav_runs(n_prev, n_next, selected, rows):
    if not rows:
        return dash.no_update
    # current index (default to first row if none selected yet)
    idx = selected[0] if selected else 0

    # which button fired?
    trig = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
    if trig == "btn-prev-run":
        new_idx = max(0, idx - 1)
    elif trig == "btn-next-run":
        new_idx = min(len(rows) - 1, idx + 1)
    else:
        return dash.no_update

    if new_idx == idx:
        return dash.no_update
    return [new_idx]

if __name__ == "__main__":
    app.run(debug=True)  # http://127.0.0.1:8050
