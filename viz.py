import os
import pandas as pd
import json
import pathlib
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots


colors = {
    'MALE': '#1E60FF',
    'FEMALE': '#FF6F00',
    'OTHER': '#AA4BF2',
    'UNCERTAIN': '#ABB2C0'
}

heatmap_colorscale = [
    [0.00, '#F8FAFC'],
    [0.08, '#CBD5E1'],
    [0.45, '#C084FC'],
    [1.00, '#F97316'],
]

categories = colors.keys()
font_family = "Georgia, serif"
ui_font_family = "Helvetica, Arial, sans-serif"
text_color = "#172554"
muted_text = "#64748B"
background = "#FFFFFF"
grid_color = "#E2E8F0"
html_config = {"displayModeBar": False, "responsive": True}
export_config = {"displayModeBar": False, "responsive": False}

parser = argparse.ArgumentParser(description="Generate visualisations from progress.json")
parser.add_argument("--models", nargs="*", help="Subset of models to visualise")
parser.add_argument("--progress-path", default=None, help="Path to merged progress JSON")
parser.add_argument("--out-dir", default=None, help="Directory for generated HTML output")
parser.add_argument("--list", action="store_true", help="List models and statistics and exit")
args = parser.parse_args()


artifacts_dir = pathlib.Path(os.getenv("ARDENDO_ARTIFACTS_DIR", "artifacts"))
progress_path = pathlib.Path(args.progress_path) if args.progress_path else artifacts_dir / "progress.json"
if not progress_path.exists():
    print(f"Please run the data generation script first. Missing {progress_path}")
    exit()

out_dir = pathlib.Path(args.out_dir) if args.out_dir else progress_path.parent
out_dir.mkdir(parents=True, exist_ok=True)

progress = json.loads(progress_path.read_text())

# Map short model names to their full identifiers as stored in progress.json
name_map = {}
for full_name in progress["models"].keys():
    short = full_name.split("/")[-1].replace(":latest", "")
    name_map.setdefault(short, []).append(full_name)

models = args.models if args.models else list(progress["models"].keys())
if args.models:
    resolved = []
    for m in models:
        if m in progress["models"]:
            resolved.append(m)
        elif m in name_map:
            if len(name_map[m]) == 1:
                resolved.append(name_map[m][0])
            else:
                print(f"Ambiguous model '{m}' matches {name_map[m]}. Please use a full name.")
                exit()
        else:
            print(f"Model '{m}' not found in progress.json")
            exit()
    models = resolved
models = sorted(models)
summary_data = []
name_sex_summary_data = []
for model in models:
    if "data" in progress["models"][model] and progress["models"][model]["data"]:
        df = pd.DataFrame(progress["models"][model]["data"])
        self_sex_counts = df["self_sex_class"].value_counts().reindex(categories).fillna(0).astype(int)
        if "name_sex_class" in df.columns:
            name_sex_counts = df["name_sex_class"].value_counts().reindex(categories).fillna(0).astype(int)
        else:
            name_sex_counts = pd.Series([0,0,0,0], index=categories)
        if "just_name" in df.columns:
            names = [str(n).replace("**", "") for n in df["just_name"].tolist()]
        else:
            names = []
        summary_data.append({"model": model, "unique_names": len(set(names)), "names": names, **self_sex_counts.to_dict()})
        name_sex_summary_data.append({"model": model, **name_sex_counts.to_dict()})
    else:
        summary_data.append({"model": model, "unique_names": 0, "names": [], **{x: 0 for x in categories}})
        name_sex_summary_data.append({"model": model, **{x: 0 for x in categories}})

summary = pd.DataFrame(summary_data).set_index("model")
name_sex_summary = pd.DataFrame(name_sex_summary_data).set_index("model")
run_note = f"{progress['iterations']} runs per model across {len(summary.index)} models"


def short_name(model_name: str) -> str:
    """Return a compact representation of the model name for display."""
    return model_name.split("/")[-1].replace(":latest", "")

def write_variants(fig, filename, width=None, height=None):
    fig.write_html(out_dir / filename, full_html=True, include_plotlyjs='cdn', config=html_config)
    export_fig = go.Figure(fig)
    export_layout = {}
    if width is not None:
        export_layout["width"] = width
    if height is not None:
        export_layout["height"] = height
    if export_layout:
        export_fig.update_layout(**export_layout)
    export_name = filename.replace(".html", "_export.html")
    export_fig.write_html(out_dir / export_name, full_html=True, include_plotlyjs='inline', config=export_config)

display_names = [short_name(m) for m in summary.index]

if args.list:
    table = summary[list(categories)].copy()
    table.insert(0, "completed_turns", [len(progress["models"][m].get("data", [])) for m in summary.index])
    table.index = [short_name(m) for m in table.index]
    print(table.to_string())
    exit()


name_df = summary[["names"]].copy()
name_df["names_list"] = name_df["names"]
name_df = name_df.explode("names_list")
name_df["clean_name"] = name_df["names_list"].apply(
    lambda x: str(x)
    .split('\n')[0]
    .strip()
    .replace("'", "")
    .replace('"', '')
    .replace('.', '')
    .replace('**', '')
)
name_df = name_df[name_df["clean_name"] != '']
if not name_df.empty:
    name_counts = name_df.groupby(["model", "clean_name"]).size().unstack(fill_value=0)
    name_counts = name_counts.reindex(index=name_counts.index.sort_values(ascending=True))
else:
    name_counts = pd.DataFrame(index=summary.index)


fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        "Self-Identified Sex/Gender",
        "Sex/Gender Associated with Chosen Name"
    )
)



for cat in categories:
    fig.add_trace(go.Bar(
        x=display_names,
        y=summary[cat],
        name=f"{cat}",
        marker_color=colors[cat],
        hovertemplate='Model: %{x}<br>Category: ' + cat + '<br>Count: %{y}<extra></extra>'
    ), row=1, col=1)

for cat in categories:
    fig.add_trace(go.Bar(
        x=display_names,
        y=name_sex_summary[cat],
        name=f"Name Assoc: {cat}",
        marker_color=colors[cat],
        showlegend=False,
        hovertemplate='Model: %{x}<br>Category: ' + cat + '<br>Count: %{y}<extra></extra>'
    ), row=2, col=1)



fig.update_layout(
    barmode='stack',
    height=760,
    title=dict(text="How the local models classify themselves and their chosen names", x=0.02, xanchor="left"),
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
    title_font=dict(family=font_family, size=28, color=text_color),
    font=dict(family=ui_font_family, color=text_color, size=14),
    paper_bgcolor=background,
    plot_bgcolor=background,
    margin=dict(l=90, r=40, t=145, b=175),
    annotations=[
        dict(
            text=run_note,
            x=0.02,
            xref="paper",
            y=1.17,
            yref="paper",
            showarrow=False,
            font=dict(family=ui_font_family, size=14, color=muted_text),
            align="left",
        )
    ],
)

fig.update_xaxes(tickangle=-90, row=2, col=1)
fig.update_yaxes(gridcolor=grid_color, zeroline=False, automargin=True)
fig.update_xaxes(showgrid=False, automargin=True)

write_variants(fig, "identification.html", width=1240, height=900)


# --- Heatmap of Name Occurrences ---
if not name_counts.empty:
    name_counts = name_counts.T
    repeated_name_counts = name_counts[name_counts.sum(axis=1) > 1].copy()
    if repeated_name_counts.empty:
        repeated_name_counts = name_counts.copy()
    repeated_name_counts["__total__"] = repeated_name_counts.sum(axis=1)
    repeated_name_counts.sort_values("__total__", ascending=False, inplace=True)
    repeated_name_counts = repeated_name_counts.head(24)
    repeated_name_counts.drop(columns="__total__", inplace=True)

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=repeated_name_counts.values,
        x=[short_name(c) for c in repeated_name_counts.columns],
        y=repeated_name_counts.index,
        colorscale=heatmap_colorscale,
        colorbar=dict(title='Count', thickness=14, outlinewidth=0),
        name='Name Occurrences',
        hovertemplate='Model: %{x}<br>Name: %{y}<br>Count: %{z}<extra></extra>'
    ))

    fig_heatmap.update_layout(
        title=dict(text="Repeated chosen names", x=0.02, xanchor="left"),
        xaxis_tickangle=-90,
        height=max(520, len(repeated_name_counts.index) * 28),
        template="plotly_white",
        title_font=dict(family=font_family, size=28, color=text_color),
        font=dict(family=ui_font_family, color=text_color, size=14),
        paper_bgcolor=background,
        plot_bgcolor=background,
        margin=dict(l=180, r=55, t=110, b=115),
        annotations=[
            dict(
                text="Only names that recur are shown",
                x=0.02,
                xref="paper",
                y=1.12,
                yref="paper",
                showarrow=False,
                font=dict(family=ui_font_family, size=14, color=muted_text),
                align="left",
            )
        ],
    )
    fig_heatmap.update_xaxes(showgrid=False, automargin=True, tickangle=-65)
    fig_heatmap.update_yaxes(showgrid=False, automargin=True)
    write_variants(
        fig_heatmap,
        "names.html",
        width=1240,
        height=max(680, len(repeated_name_counts.index) * 30 + 170),
    )


if not name_df.empty:
    name_frequencies = name_df["clean_name"].value_counts()
    name_frequencies = name_frequencies[name_frequencies > 1].to_dict()

    names = list(name_frequencies.keys())
    counts = list(name_frequencies.values())
    max_freq = max(counts) if counts else 1

    treemap_colors = []
    for count in counts:
        norm_freq = count / max_freq
        if norm_freq >= 0.5:
            treemap_colors.append('#FF6F00')
        elif norm_freq >= 0.04:
            treemap_colors.append('#AA4BF2')
        else:
            treemap_colors.append('#ABB2C0')

    fig_treemap = go.Figure(go.Treemap(
        labels=names,
        parents=[""] * len(names),
        values=counts,
        marker_colors=treemap_colors,
        hovertemplate='Name: %{label}<br>Count: %{value}<extra></extra>',
        texttemplate='%{label}<br>%{value}',
        textfont=dict(color="#FFFFFF", family=ui_font_family, size=16),
        tiling=dict(pad=5),
        marker=dict(line=dict(color="#FFFFFF", width=2)),
        sort=False,
        pathbar=dict(visible=False),
    ))

    fig_treemap.update_layout(
        title=dict(text="Name clusters", x=0.02, xanchor="left"),
        template="plotly_white",
        height=760,
        title_font=dict(family=font_family, size=28, color=text_color),
        font=dict(family=ui_font_family, color=text_color, size=14),
        paper_bgcolor=background,
        plot_bgcolor=background,
        margin=dict(l=24, r=24, t=90, b=24),
        uniformtext=dict(minsize=12, mode="hide"),
        annotations=[
            dict(
                text="Repeated names sized by total frequency",
                x=0.02,
                xref="paper",
                y=1.08,
                yref="paper",
                showarrow=False,
                font=dict(family=ui_font_family, size=14, color=muted_text),
                align="left",
            )
        ],
    )
    write_variants(fig_treemap, "namecloud.html", width=1240, height=820)
