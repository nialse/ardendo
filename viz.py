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
turn_counts = pd.Series(
    {model: len(progress["models"][model].get("data", [])) for model in summary.index},
    index=summary.index
)
summary_share = summary[list(categories)].div(turn_counts.replace(0, pd.NA), axis=0).fillna(0) * 100
name_sex_share = name_sex_summary[list(categories)].div(turn_counts.replace(0, pd.NA), axis=0).fillna(0) * 100
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
        y=summary_share[cat],
        customdata=summary[cat],
        name=f"{cat}",
        marker_color=colors[cat],
        hovertemplate='Model: %{x}<br>Category: ' + cat + '<br>Share: %{y:.1f}%<br>Count: %{customdata}<extra></extra>'
    ), row=1, col=1)

for cat in categories:
    fig.add_trace(go.Bar(
        x=display_names,
        y=name_sex_share[cat],
        customdata=name_sex_summary[cat],
        name=f"Name Assoc: {cat}",
        marker_color=colors[cat],
        showlegend=False,
        hovertemplate='Model: %{x}<br>Category: ' + cat + '<br>Share: %{y:.1f}%<br>Count: %{customdata}<extra></extra>'
    ), row=2, col=1)



fig.update_layout(
    barmode='stack',
    height=760,
    title=dict(
        text=(
            "How models classify themselves and their chosen names"
            f"<br><span style='font-size:14px;color:{muted_text};'>{run_note}. "
            "Bars show share of responses, not raw totals.</span>"
        ),
        x=0.02,
        xanchor="left",
        y=0.97,
        pad=dict(b=16),
    ),
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
    title_font=dict(family=font_family, size=28, color=text_color),
    font=dict(family=ui_font_family, color=text_color, size=14),
    paper_bgcolor=background,
    plot_bgcolor=background,
    margin=dict(l=90, r=40, t=135, b=175),
)

fig.update_xaxes(tickangle=-90, row=2, col=1)
fig.update_yaxes(gridcolor=grid_color, zeroline=False, automargin=True)
fig.update_yaxes(range=[0, 100], ticksuffix="%", row=1, col=1)
fig.update_yaxes(range=[0, 100], ticksuffix="%", row=2, col=1)
fig.update_xaxes(showgrid=False, automargin=True)

write_variants(fig, "identification.html", width=1240, height=900)


if not name_counts.empty:
    name_counts = name_counts.T
    repeated_name_counts = name_counts[name_counts.sum(axis=1) > 1].copy()
    if repeated_name_counts.empty:
        repeated_name_counts = name_counts.copy()
    repeated_name_counts["__total__"] = repeated_name_counts.sum(axis=1)
    repeated_name_counts.sort_values("__total__", ascending=False, inplace=True)
    repeated_name_counts = repeated_name_counts.head(18)
    repeated_name_totals = repeated_name_counts["__total__"].copy()
    repeated_name_counts.drop(columns="__total__", inplace=True)

    dot_rows = []
    for name in repeated_name_counts.index:
        for model in repeated_name_counts.columns:
            count = int(repeated_name_counts.loc[name, model])
            if count > 0:
                completed = int(turn_counts[model]) if int(turn_counts[model]) else 0
                share = (count / completed) * 100 if completed else 0
                dot_rows.append(
                    {
                        "model": short_name(model),
                        "name": name,
                        "count": count,
                        "share": share,
                        "total": int(repeated_name_totals[name]),
                    }
                )

    dot_df = pd.DataFrame(dot_rows)
    size_max = dot_df["count"].max() if not dot_df.empty else 1

    fig_heatmap = go.Figure(
        go.Scatter(
            x=dot_df["model"],
            y=dot_df["name"],
            mode="markers",
            marker=dict(
                size=dot_df["count"],
                sizemode="area",
                sizeref=(2.0 * size_max) / (34 ** 2) if size_max else 1,
                sizemin=8,
                color=dot_df["share"],
                colorscale=heatmap_colorscale,
                cmin=0,
                cmax=100,
                line=dict(color="#FFFFFF", width=1.5),
                colorbar=dict(title='Share of model runs', thickness=14, outlinewidth=0, ticksuffix='%'),
            ),
            customdata=dot_df[["count", "share", "total"]],
            hovertemplate='Model: %{x}<br>Name: %{y}<br>Count: %{customdata[0]}<br>Share of model runs: %{customdata[1]:.1f}%<br>Total across run: %{customdata[2]}<extra></extra>'
        )
    )

    fig_heatmap.update_layout(
        title=dict(text="Repeated chosen names by model", x=0.02, xanchor="left"),
        xaxis_tickangle=-90,
        height=max(560, len(repeated_name_counts.index) * 32),
        template="plotly_white",
        title_font=dict(family=font_family, size=28, color=text_color),
        font=dict(family=ui_font_family, color=text_color, size=14),
        paper_bgcolor=background,
        plot_bgcolor=background,
        margin=dict(l=210, r=95, t=115, b=130),
        annotations=[
            dict(
                text="Only names repeated across the run are shown. Dot area = count, color = share of that model's runs.",
                x=0.02,
                xref="paper",
                y=1.10,
                yref="paper",
                showarrow=False,
                font=dict(family=ui_font_family, size=14, color=muted_text),
                align="left",
            )
        ],
    )
    fig_heatmap.update_xaxes(showgrid=False, automargin=True, tickangle=-65)
    fig_heatmap.update_yaxes(showgrid=False, automargin=True, categoryorder='array', categoryarray=list(repeated_name_counts.index[::-1]))
    write_variants(
        fig_heatmap,
        "names.html",
        width=1240,
        height=max(720, len(repeated_name_counts.index) * 34 + 170),
    )


if not name_df.empty:
    name_frequencies = name_df["clean_name"].value_counts()
    repeated_name_frequencies = name_frequencies[name_frequencies > 1]
    if repeated_name_frequencies.empty:
        repeated_name_frequencies = name_frequencies
    top_names = repeated_name_frequencies.head(14).sort_values(ascending=True)
    cumulative_share = (name_frequencies.sort_values(ascending=False).cumsum() / name_frequencies.sum()) * 100

    fig_treemap = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.12,
    )

    fig_treemap.add_trace(
        go.Bar(
            x=top_names.values,
            y=top_names.index,
            orientation='h',
            marker_color=['#AA4BF2'] * max(len(top_names) - 3, 0) + ['#F97316'] * min(len(top_names), 3),
            text=top_names.values,
            textposition='outside',
            cliponaxis=False,
            hovertemplate='Name: %{y}<br>Count: %{x}<extra></extra>'
        ),
        row=1,
        col=1
    )

    fig_treemap.add_trace(
        go.Scatter(
            x=cumulative_share.values,
            y=list(range(1, len(cumulative_share) + 1)),
            mode='lines+markers',
            line=dict(color='#AA4BF2', width=4),
            marker=dict(size=7, color='#F97316'),
            fill='tozerox',
            fillcolor='rgba(170, 75, 242, 0.12)',
            hovertemplate='Cumulative share: %{x:.1f}%<br>Name rank: %{y}<extra></extra>'
        ),
        row=1,
        col=2
    )

    fig_treemap.update_layout(
        title=dict(text="Name concentration", x=0.02, xanchor="left", y=0.96),
        template="plotly_white",
        height=760,
        title_font=dict(family=font_family, size=28, color=text_color),
        font=dict(family=ui_font_family, color=text_color, size=14),
        paper_bgcolor=background,
        plot_bgcolor=background,
        margin=dict(l=120, r=70, t=140, b=80),
        showlegend=False,
    )

    fig_treemap.update_xaxes(
        title_text="Occurrences",
        gridcolor=grid_color,
        zeroline=False,
        automargin=True,
        row=1,
        col=1
    )
    fig_treemap.update_yaxes(
        showgrid=False,
        automargin=True,
        row=1,
        col=1
    )
    fig_treemap.update_xaxes(
        title_text="Cumulative share",
        gridcolor=grid_color,
        zeroline=False,
        ticksuffix="%",
        range=[0, 100],
        automargin=True,
        row=1,
        col=2
    )
    fig_treemap.update_yaxes(title_text="Name rank", gridcolor=grid_color, zeroline=False, automargin=True, row=1, col=2)
    write_variants(fig_treemap, "namecloud.html", width=1240, height=820)
