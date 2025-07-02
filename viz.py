import pandas as pd
import json
import pathlib
import re
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
    [0.00, '#FFFFFF'],
    [0.04, '#ABB2C0'],
    [0.5, '#AA4BF2'],
    [1.00, '#FF6F00'],
]

categories = colors.keys()

parser = argparse.ArgumentParser(description="Generate visualisations from progress.json")
parser.add_argument("--models", nargs="*", help="Subset of models to visualise")
parser.add_argument("--list", action="store_true", help="List models and statistics and exit")
args = parser.parse_args()


progress_path = pathlib.Path("progress.json")
if not progress_path.exists():
    print("Please run the data generation script first.")
    exit()
    
progress = json.loads(progress_path.read_text())

models = args.models if args.models else list(progress["models"].keys())
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
        names = df["just_name"].tolist() if "just_name" in df.columns else []
        summary_data.append({"model": model, "unique_names": len(set(names)), "names": names, **self_sex_counts.to_dict()})
        name_sex_summary_data.append({"model": model, **name_sex_counts.to_dict()})
    else:
        summary_data.append({"model": model, "unique_names": 0, "names": [], **{x: 0 for x in categories}})
        name_sex_summary_data.append({"model": model, **{x: 0 for x in categories}})

summary = pd.DataFrame(summary_data).set_index("model")
name_sex_summary = pd.DataFrame(name_sex_summary_data).set_index("model")

summary.index = summary.index.str.replace(':latest', '', regex=False)
name_sex_summary.index = name_sex_summary.index.str.replace(':latest', '', regex=False)

if args.list:
    table = summary[list(categories)].copy()
    table.insert(0, "completed_turns", [len(progress["models"][m].get("data", [])) for m in summary.index])
    print(table.to_string())
    exit()


name_df = summary[["names"]].copy()
name_df["names_list"] = name_df["names"]
name_df = name_df.explode("names_list")
name_df["clean_name"] = name_df["names_list"].apply(lambda x: str(x).split('\n')[0].strip().replace("'", "").replace('"', '').replace('.', ''))
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
        x=summary.index,
        y=summary[cat],
        name=f"{cat}",
        marker_color=colors[cat],
        hovertemplate='Model: %{x}<br>Category: ' + cat + '<br>Count: %{y}<extra></extra>'
    ), row=1, col=1)

for cat in categories:
    fig.add_trace(go.Bar(
        x=name_sex_summary.index,
        y=name_sex_summary[cat],
        name=f"Name Assoc: {cat}",
        marker_color=colors[cat],
        showlegend=False,
        hovertemplate='Model: %{x}<br>Category: ' + cat + '<br>Count: %{y}<extra></extra>'
    ), row=2, col=1)



fig.update_layout(
    barmode='stack',
    height=800,
    title_x=0.5,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    title_font=dict(family="Helvetica"),
    font=dict(family="Helvetica"),
)

fig.update_xaxes(tickangle=-90, row=2, col=1)

fig.write_html("identification.html", full_html=True, include_plotlyjs='cdn')


# --- Heatmap of Name Occurrences ---
if not name_counts.empty:
    # Transpose and sort for better visualization (models on x-axis, names on y-axis)
    name_counts = name_counts.T
    name_counts.sort_index(inplace=True, ascending=False)

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=name_counts.values,
        x=name_counts.columns,
        y=name_counts.index,
        colorscale=heatmap_colorscale,
        colorbar=dict(title='Count'),
        name='Name Occurrences',
        hovertemplate='Model: %{x}<br>Name: %{y}<br>Count: %{z}<extra></extra>'
    ))

    fig_heatmap.update_layout(
        title_text="<b>Name Occurrences per Model</b>",
        title_x=0.5,
        xaxis_tickangle=-90,
        height=max(600, len(name_counts.index) * 20),
        template="plotly_white",
        title_font=dict(family="Helvetica"),
        font=dict(family="Helvetica"),
    )
    fig_heatmap.write_html("names.html", full_html=True, include_plotlyjs='cdn')


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
        textinfo='label+value'
    ))

    fig_treemap.update_layout(
        title_x=0.5,
        template="plotly_white",
        height=800,
        title_font=dict(family="Helvetica"),
        font=dict(family="Helvetica"),
    )
    fig_treemap.write_html("namecloud.html", full_html=True, include_plotlyjs='cdn')

