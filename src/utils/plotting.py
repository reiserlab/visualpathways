# %%
"""Plotting functions grouped by legacy notebook (1..11).

Each section hosts plot_* functions for the corresponding figure group.
Functions consume cached artefacts (loaded via `preprocessing.py`) and write
figures to `FIG_DIR / <section>/`.
"""
import re
from pathlib import Path

# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# %%
from utils.config import DATA_DIR, FIG_DATASET as DATASET, FIG_DIR, HTML_FIG_DIR, SIDE_CHAR


# %%
_WRITE_HTML = False


def set_html_output(enabled: bool) -> None:
    """Enable or disable optional HTML sidecar outputs."""
    global _WRITE_HTML
    _WRITE_HTML = bool(enabled)


# %%
_SIDE_WORD = {"r": "right", "l": "left"}


# %% [markdown]
# === 0. Shared plot helpers ===


# %%
def _pie(df: pd.DataFrame, val: str, col: str, cmap: dict, *, height=400, width=400, title=""):
    fig = px.pie(df, values=val, names=col, title=title, color=col, color_discrete_map=cmap)
    fig.update_traces(textposition="inside", texttemplate="%{percent:.0%}")
    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="arial", size=18),
    )
    return fig


# %%
def _save(fig, save_dir: Path | None, filename: str, *, html: bool = False, rasterize: bool = False):
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    if rasterize and filename.endswith('.pdf'):
        from io import BytesIO
        from PIL import Image
        png_bytes = fig.to_image(format='png', scale=4)
        Image.open(BytesIO(png_bytes)).save(save_dir / filename, 'PDF', resolution=300)
    else:
        fig.write_image(save_dir / filename)
    if html and _WRITE_HTML:
        stem = Path(filename).stem
        fig.write_html(save_dir / f"{stem}.html")
        HTML_FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(HTML_FIG_DIR / f"{stem}.html")


# %%
_AXIS = dict(
    showline=True, linewidth=1, linecolor="black", mirror=True,
    ticks="outside", ticklen=5, tickcolor="black",
)


# %%
def _hist_lines(df, val, col, bins, color_df, *, height=400, width=500, norm=True):
    fig = go.Figure()
    centers = (bins[:-1] + bins[1:]) / 2
    for group in df[col].unique():
        sub = df[df[col] == group]
        counts, _ = np.histogram(sub[val], bins=bins)
        if norm:
            counts = counts / counts.sum()
        fig.add_trace(go.Scatter(
            x=centers, y=counts, mode="markers+lines", name=str(group),
            marker=dict(color=color_df.loc[group, "color"]),
            line=dict(color=color_df.loc[group, "color"]),
        ))
    fig.update_layout(
        font=dict(family="arial", size=18), xaxis_title=val,
        xaxis_range=[bins[0], bins[-1]],
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width,
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _hist_lines_v(df, val, col, bins, color_df, *, height=400, width=500, norm=True):
    """Vertical variant of `_hist_lines`: x-axis = fraction, y-axis = value."""
    fig = go.Figure()
    centers = (bins[:-1] + bins[1:]) / 2
    for group in df[col].unique():
        sub = df[df[col] == group]
        counts, _ = np.histogram(sub[val], bins=bins)
        if norm:
            counts = counts / counts.sum()
        fig.add_trace(go.Scatter(
            x=counts, y=centers, mode="markers+lines", name=str(group),
            marker=dict(color=color_df.loc[group, "color"]),
            line=dict(color=color_df.loc[group, "color"]),
        ))
    fig.update_layout(
        font=dict(family="arial", size=18), yaxis_title=val,
        yaxis_range=[bins[0], bins[-1]],
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width,
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _hist_cumsum_lines(df, val, col, bins, color_df, *, height=500, width=400):
    """Cumulative-sum histogram by group, horizontal orientation."""
    fig = go.Figure()
    centers = (bins[:-1] + bins[1:]) / 2
    for group in df[col].unique():
        sub = df[df[col] == group]
        counts, _ = np.histogram(sub[val], bins=bins)
        if counts.sum() == 0:
            continue
        cum = np.cumsum(counts / counts.sum())
        fig.add_trace(go.Scatter(
            x=centers, y=cum, mode="markers+lines", name=str(group),
            marker=dict(color=color_df.loc[group, "color"]),
            line=dict(color=color_df.loc[group, "color"]),
        ))
    fig.update_layout(
        font=dict(family="arial", size=18),
        xaxis_title=val, yaxis_title="cum. frac.",
        xaxis_range=[bins[0], bins[-1]],
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width,
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _hist_cumsum_lines_v(df, val, col, bins, color_df, *, height=500, width=400):
    """Vertical cumulative-sum histogram."""
    fig = go.Figure()
    centers = (bins[:-1] + bins[1:]) / 2
    for group in df[col].unique():
        sub = df[df[col] == group]
        counts, _ = np.histogram(sub[val], bins=bins)
        if counts.sum() == 0:
            continue
        cum = np.cumsum(counts / counts.sum())
        fig.add_trace(go.Scatter(
            x=cum, y=centers, mode="markers+lines", name=str(group),
            marker=dict(color=color_df.loc[group, "color"]),
            line=dict(color=color_df.loc[group, "color"]),
        ))
    fig.update_layout(
        font=dict(family="arial", size=18),
        xaxis_title="cum. frac.", yaxis_title=val,
        yaxis_range=[bins[0], bins[-1]],
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width,
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _hist_sum_lines(df, val, sum_val, col, bins, color_df, *, height=400, width=500, norm=True):
    centers = (bins[:-1] + bins[1:]) / 2
    plot_df = df.copy()
    plot_df["val_bin"] = pd.cut(plot_df[val], bins=bins, labels=centers).astype(float)
    fig = go.Figure()
    for group in plot_df[col].unique():
        sub = plot_df[plot_df[col] == group]
        binned = sub.groupby("val_bin", observed=False)[sum_val].sum()
        binned = binned.reindex(centers, fill_value=0).reset_index()
        binned.columns = ["bin", "count"]
        if norm:
            binned["count"] = binned["count"] / binned["count"].sum()
        fig.add_trace(go.Scatter(
            x=binned["bin"], y=binned["count"], mode="markers+lines", name=str(group),
            marker=dict(color=color_df.loc[group, "color"]),
            line=dict(color=color_df.loc[group, "color"]),
        ))
    fig.update_layout(
        font=dict(family="arial", size=18), xaxis_title=val,
        xaxis_range=[bins[0], bins[-1]],
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width,
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _scatter(plot_df, x_col, y_col, c_col, t_col, color_df, t_stars=(), *,
             height=400, width=400, marker="circle", size=7, show_labels=True):
    fig = go.Figure(go.Scatter(
        x=plot_df[x_col].values, y=plot_df[y_col].values, mode="markers",
        marker=dict(
            size=size,
            color=color_df.loc[plot_df[c_col].values, "color"].values,
            opacity=0.4, symbol=marker,
        ),
        text=plot_df[t_col],
    ))
    for t in t_stars:
        sub = plot_df[plot_df[t_col] == t]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[x_col].values, y=sub[y_col].values,
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=size,
                color=color_df.loc[sub[c_col].values, "color"].values,
                opacity=1, symbol=marker, line=dict(color="black", width=1),
            ),
            text=[s.replace("_R", "") for s in sub[t_col]],
            textposition="bottom center",
        ))
    fig.update_layout(
        xaxis_title=x_col, yaxis_title=y_col, width=width, height=height,
        font=dict(family="arial", size=18), showlegend=False,
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _box(plot_df, x_col, y_col, c_col, cmap, *, height=600, width=400):
    fig = px.box(plot_df, x=x_col, y=y_col, points="all",
                 color=c_col, color_discrete_map=cmap)
    fig.update_layout(
        font=dict(family="arial", size=18),
        xaxis_title=x_col, yaxis_title=y_col,
        xaxis_range=[0.9 * plot_df[x_col].min(), 1.1 * plot_df[x_col].max()],
        yaxis_tickvals=plot_df[y_col].values,
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width, showlegend=False,
    )
    fig.update_xaxes(title_standoff=0, tickangle=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _box_scatter(plot_df, x_col, y_col, c_col, color_df, *, height=350, width=450):
    """Grouped horizontal-by-group box (one box-per-category × group). Unlike
    `_box`, does not set per-row y-ticks — used when the group dimension has
    many rows."""
    fig = go.Figure()
    for group in plot_df[c_col].unique():
        sub = plot_df[plot_df[c_col] == group]
        fig.add_trace(go.Box(
            x=sub[x_col], y=sub[y_col], name=str(group),
            marker=dict(color=color_df.loc[group, "color"]),
            line=dict(color=color_df.loc[group, "color"]),
        ))
    fig.update_layout(
        xaxis_title=x_col, yaxis_title=y_col,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="arial", size=18),
        height=height, width=width, boxmode="group",
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _heatmatrix(
    mat, tick_labels, *, cmap="Greys", height=500, width=500, anno=None,
    x_tick_labels=None, y_tick_labels=None,
):
    x_tick_labels = list(tick_labels if x_tick_labels is None else x_tick_labels)
    y_tick_labels = list(tick_labels if y_tick_labels is None else y_tick_labels)
    n_x = len(x_tick_labels)
    n_y = len(y_tick_labels)
    fig = go.Figure(data=go.Heatmap(
        z=mat, x=x_tick_labels, y=y_tick_labels,
        colorscale=cmap, reversescale=False, showscale=True,
        text=anno, texttemplate="%{text}", textfont={"size": 18},
    ))
    n_diag = min(n_x, n_y)
    fig.add_shape(type="line", x0=-0.5, y0=-0.5, x1=n_diag - 0.5, y1=n_diag - 0.5,
                  line=dict(color="gray", width=2))
    fig.update_layout(
        xaxis_title="post layer", yaxis_title="pre layer",
        xaxis_tickvals=list(range(n_x)), xaxis_ticktext=x_tick_labels,
        yaxis_tickvals=list(range(n_y)), yaxis_ticktext=y_tick_labels,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="arial", size=18),
        height=height, width=width,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _point_line(plot_df, x_col, y_col, *, x_thre=None, y_thre=None,
                height=500, width=600, color="black"):
    fig = go.Figure()
    if x_thre is not None:
        fig.add_vline(x=x_thre, line_width=1, line_dash="dash", line_color="gray")
    if y_thre is not None:
        fig.add_hline(y=y_thre, line_width=1, line_dash="dash", line_color="gray")
    fig.add_trace(go.Scatter(
        x=plot_df[x_col].values, y=plot_df[y_col].values,
        mode="markers+lines",
        marker=dict(size=8, color=color, opacity=0.7),
        line=dict(width=1, color=color),
    ))
    fig.update_layout(
        font=dict(family="arial", size=18),
        xaxis_title=x_col, yaxis_title=y_col,
        xaxis_range=[0.9 * plot_df[x_col].min(), 1.1 * plot_df[x_col].max()],
        yaxis_range=[0.9 * plot_df[y_col].min(), 1.1 * plot_df[y_col].max()],
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width, showlegend=False,
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _stacked_bars(plot_df, x_col, y_col, c_col, cmap, *, height=350, width=500, xrot=0, font_size=18):
    fig = px.bar(
        plot_df, x=x_col, y=y_col, color=c_col,
        color_discrete_map=cmap, category_orders={c_col: list(cmap.keys())},
    )
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_hline(y=1, line_color="black", line_width=1)
    fig.update_layout(
        boxmode="group", xaxis_tickangle=xrot,
        yaxis_range=[-0.01, 1.01],
        font=dict(family="arial", size=font_size),
        height=height, width=width,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, tickvals=plot_df[x_col].values)
    fig.update_yaxes(showgrid=False, tickvals=[0, 1])
    return fig


# %%
def _violin_median(
    df, val_col, cat_col, *, labels=None, height=800, width=500,
    xaxis_title="", xaxis_tickvals=None, xaxis_range=None,
):
    """Horizontal violin of `val_col` across categories in `cat_col`, with a
    median marker per category. Categories appear top-to-bottom in the order
    they first occur in `df`."""
    cats = df[cat_col].drop_duplicates().tolist()
    pos = {c: i for i, c in enumerate(cats)}
    plot_df = df.copy()
    plot_df["__y"] = plot_df[cat_col].map(pos)
    medians = (
        plot_df.groupby(cat_col, observed=False)[val_col].median()
        .reindex(cats).reset_index()
    )
    medians["__y"] = medians[cat_col].map(pos)

    fig = px.violin(plot_df, x=val_col, y="__y", orientation="h")
    fig.update_traces(line_color="gray", marker=dict(size=3, opacity=0.4, color="gray"))
    fig.add_scatter(
        x=medians[val_col], y=medians["__y"], mode="markers",
        marker=dict(color="black", size=10), showlegend=False,
    )
    fig.update_layout(
        font=dict(family="arial", size=18),
        xaxis_title=xaxis_title, yaxis_title="",
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width, showlegend=False,
    )
    ticktext = labels if labels is not None else cats
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(len(cats))), ticktext=ticktext,
        title_standoff=2, autorange="reversed", **_AXIS,
    )
    xkw = {"title_standoff": 0, **_AXIS}
    if xaxis_tickvals is not None:
        xkw["tickvals"] = xaxis_tickvals
    if xaxis_range is not None:
        xkw["range"] = xaxis_range
    fig.update_xaxes(**xkw)
    return fig


# %%
def _add_gaussian_ellipses(fig, params_df, *, example_bid=None, color="red", y_scale=1.0):
    """Overlay fitted 2D-Gaussian ellipses + centres onto an existing figure.

    `params_df` must have columns x0, y0, a, b, phi; if `example_bid` is given
    and matches `params_df['bodyId']`, that row is drawn with a thicker line.
    `y_scale` multiplies the y coordinate of every drawn point — pass
    `y_scale=np.sqrt(3)` to un-scale Gaussian fits (which were done in
    isotropic y/sqrt(3) units) back onto raw hex coords.
    """
    t = np.linspace(0, 2 * np.pi, 60)
    ct, st = np.cos(t), np.sin(t)
    fig.add_trace(go.Scatter(
        x=params_df["x0"].values, y=y_scale * params_df["y0"].values, mode="markers",
        marker=dict(color=color, size=6, symbol="cross"), showlegend=False,
    ))
    for _, row in params_df.iterrows():
        highlight = example_bid is not None and row.get("bodyId") == example_bid
        lw = 4 if highlight else 1
        cp, sp = np.cos(row["phi"]), np.sin(row["phi"])
        x = row["x0"] + row["a"] * cp * ct - row["b"] * sp * st
        y_fit = row["y0"] + row["b"] * cp * st + row["a"] * sp * ct
        fig.add_trace(go.Scatter(
            x=x, y=y_scale * y_fit, mode="lines",
            line=dict(color=color, width=lw), showlegend=False,
        ))
    return fig


# %%
def _eyesymbol(color_segments):
    """Stylised 5-segment eye pictogram: 4 outer colored wedges + colored inner
    ellipse. `color_segments`: 5 CSS color strings, ordered
    (sector1, sector2, sector3, sector4, center)."""
    a, b = 1.2, 2
    a2, b2 = 0.5, 0.8
    N = 200
    theta = np.linspace(-np.pi / 4, 7 * np.pi / 4, N)
    wedges = [
        (-np.pi / 4, np.pi / 4), (np.pi / 4, 3 * np.pi / 4),
        (3 * np.pi / 4, 5 * np.pi / 4), (5 * np.pi / 4, 7 * np.pi / 4),
    ]
    # legacy index mapping: outer wedges (R, U, L, D) draw from sectors 3, 2, 1, 4
    wedge_colors = [color_segments[2], color_segments[1], color_segments[0], color_segments[3]]
    fig = go.Figure()
    for (start, end), color in zip(wedges, wedge_colors):
        mask = (theta >= start) & (theta <= end + 2 * np.pi / N)
        x = np.concatenate(([0], a * np.cos(theta[mask]), [0]))
        y = np.concatenate(([0], b * np.sin(theta[mask]), [0]))
        fig.add_trace(go.Scatter(
            x=x, y=y, fill="toself", fillcolor=color,
            line=dict(color="black", width=1), mode="lines", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=a2 * np.cos(theta), y=b2 * np.sin(theta),
        fill="toself", fillcolor=color_segments[4],
        line=dict(color="black", width=1.5), mode="lines", showlegend=False,
    ))
    fig.update_layout(
        width=500, height=500,
        xaxis=dict(scaleanchor="y", visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white", margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


# %%
def _directedness_triangle(plot_df, c_col, t_col, cmap, t_stars=(), *, height=450, width=450):
    fig = px.scatter_ternary(
        plot_df, a="frac_la", b="frac_fb", c="frac_ff", hover_name=t_col,
        color=c_col, color_discrete_map=cmap,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.4))
    for t in t_stars:
        sub = plot_df[plot_df[t_col] == t]
        if sub.empty:
            continue
        fig.add_trace(go.Scatterternary(
            a=sub["frac_la"].values, b=sub["frac_fb"].values, c=sub["frac_ff"].values,
            mode="markers+text",
            marker=dict(size=7, color=cmap[sub[c_col].values[0]], opacity=1,
                        line=dict(color="black", width=1)),
            text=[s.split("_")[0] for s in sub[t_col]],
            textposition="middle left",
        ))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="arial", size=18),
        ternary=dict(
            sum=1,
            aaxis=dict(title="la", showgrid=False, showline=True, linecolor="black", tickvals=[0, 0.5, 1]),
            baxis=dict(title="fb", showgrid=False, showline=True, linecolor="black", tickvals=[0, 0.5, 1]),
            caxis=dict(title="ff", showgrid=False, showline=True, linecolor="black", tickvals=[0, 0.5, 1]),
        ),
        height=height, width=width, showlegend=False,
    )
    return fig


# %%
def _attach_flow_layers(paths_df: pd.DataFrame, flow_type: pd.DataFrame) -> pd.DataFrame:
    """Merge per-instance hitting times onto pre/post, then strip `_R`/`_L`."""
    out = paths_df.merge(
        flow_type[['instance', 'hitting_time']], how='left',
        left_on='pre', right_on='instance',
    ).rename(columns={'hitting_time': 'pre_layer'}).drop(columns=['instance'])
    out = out.merge(
        flow_type[['instance', 'hitting_time']], how='left',
        left_on='post', right_on='instance',
    ).rename(columns={'hitting_time': 'post_layer'}).drop(columns=['instance'])
    out['pre'] = [s[:-2] for s in out['pre']]
    out['post'] = [s[:-2] for s in out['post']]
    return out


# %%
def plot_pathway(
    paths_dict: dict, instance: str, flow_type: pd.DataFrame,
    *,
    thre_cumsum: float = 0.4, thre_step_min: float = 0.0,
    top_n: int | None = None,
    neuron_to_color: dict | None = None,
    neuron_to_sign: dict | None = None, sign_color_map: dict | None = None,
    figsize: tuple | None = None,
    frac_text_pos: tuple = (0.85, 0.8), frac_text_size: int = 18,
    node_size: int = 1000, node_text_size: int = 18,
    edge_text: bool = False, edge_text_size: int = 10,
    weight_decimals: int = 2, highlight: bool = True,
    save_path: Path | None = None,
    show: bool = True,
    show_frac: bool = True,
):
    """Render an example-neuron pathway diagram via `plot_paths`.

    `paths_dict` is the output of `pre.get_paths_to_instance` /
    `pre.get_participation_paths` / `pre.get_cb_paths_to_instance`.
    Filter thresholds are applied here (not cached). Returns the rendered
    `conn_paths_frac` (0 if no surviving paths).

    When `top_n` is set, selection switches to the strongest `top_n` paths
    (via `filter_all_paths_to_top_n`) instead of the `thre_cumsum`
    cumulative-weight set; `thre_cumsum` is ignored in that mode while
    `thre_step_min` still applies.
    """
    from connectome_interpreter.utils import plot_paths
    import matplotlib.pyplot as plt

    all_paths = paths_dict['paths']
    hit_inst = paths_dict['hit_inst']
    if not all_paths:
        print(f'no paths for {instance}')
        return 0.0

    if figsize is None:
        figsize = (1.5 + hit_inst * 3, 8)

    if top_n is not None:
        from utils.path_filtering import filter_all_paths_to_top_n
        filtered, w_filter, w_all, _ = filter_all_paths_to_top_n(
            all_paths, top_n=top_n, thre_step_min=thre_step_min,
        )
    else:
        from utils.path_filtering import filter_all_paths_to_top_set
        filtered, w_filter, w_all, _ = filter_all_paths_to_top_set(
            all_paths, thre_cumsum=thre_cumsum, thre_step_min=thre_step_min,
        )
    conn_paths_frac = w_filter / w_all if w_all > 0 else 0.0

    if isinstance(filtered, list):
        dfs = [p for p in filtered if p is not None and not p.empty]
        if not dfs:
            print(f'no surviving paths for {instance} after filter')
            return conn_paths_frac
        paths_df = pd.concat(dfs, axis=0)
    else:
        if filtered is None or filtered.empty:
            print(f'no surviving paths for {instance} after filter')
            return conn_paths_frac
        paths_df = filtered

    paths_df = paths_df[['pre', 'post', 'weight']].drop_duplicates()
    paths_df = paths_df[paths_df['pre'] != paths_df['post']]
    paths_df = _attach_flow_layers(paths_df, flow_type)

    highlight_nodes = [instance[:-2]] if highlight else []
    # Pass show=False so plot_paths returns (fig, ax) instead of calling
    # plt.show() — otherwise the inline backend clears pyplot's current
    # figure and the text/savefig below land on a fresh empty figure.
    fig, ax = plot_paths(
        paths_df,
        neuron_to_color=neuron_to_color,
        neuron_to_sign=neuron_to_sign, sign_color_map=sign_color_map,
        interactive=False, node_size=node_size, node_text_size=node_text_size,
        weight_decimals=weight_decimals, edge_text=edge_text, edge_text_size=edge_text_size,
        highlight_nodes=highlight_nodes, save_plot=False, figsize=figsize,
        show=False,
    )
    if show_frac:
        ax.text(
            frac_text_pos[0], frac_text_pos[1], rf'$\mathit{{f}}$ = {conn_paths_frac: .2f}',
            transform=ax.transAxes, fontsize=frac_text_size,
        )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(save_path), bbox_inches='tight', dpi=300,
            format='pdf', transparent=True,
        )
    if show:
        plt.show()
    plt.close(fig)
    
    return conn_paths_frac, fig


# %%
def plot_comparison_scatter(
    df: pd.DataFrame, x_col: str, y_col: str,
    color_col: str, color_df: pd.DataFrame, *,
    xaxis_title: str, yaxis_title: str, filename_stem: str,
    save_dir: Path | None = None, stars: tuple = (), id_col: str = "instance",
    size: int = 12, width: int = 600, height: int = 600,
    diag_range: tuple | list | None = None, axis_range: tuple | list | None = None,
    log_axes: bool = False,
):
    """Scatter of a quantity from two models (x vs y), colored by `color_col`,
    with optional identity diagonal, log axes, and shared axis range."""
    fig = _scatter(
        df, x_col, y_col, color_col, id_col, color_df, stars,
        height=height, width=width, size=size,
    )
    if diag_range is not None:
        fig.add_trace(go.Scatter(
            x=list(diag_range), y=list(diag_range), mode="lines",
            line=dict(color="gray", width=2, dash="dash"), showlegend=False,
        ))
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    if log_axes:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    if axis_range is not None:
        fig.update_xaxes(range=list(axis_range))
        fig.update_yaxes(range=list(axis_range))
    _save(fig, save_dir, f"{filename_stem}.pdf")
    return fig


# %% [markdown]
# === 1. Inventory ===


# %%
def plot_inventory_full_brain(
    meta: pd.DataFrame,
    colors: dict,
    *,
    dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """Pie charts of neurons and cell types by region (full brain minus left OL).

    `colors`: {'region': series/dict of color by region}.
    Returns list of (fig, filename) pairs.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "inventory"
    m_all = meta[meta.region != "other"]
    m = meta[(meta.region != "other") & (meta.region != "left OL")]
    cmap = colors["region"]
    figs = []
    count_df = m_all.groupby("region")["bodyId"].nunique().to_frame(name="count").reset_index()
    fig = _pie(count_df, "count", "region", cmap, height=350, width=350)
    fig.update_layout(showlegend=False)
    figs.append((fig, f"{dataset}_neuron_region_pie.pdf"))

    count_type_df = m_all.groupby("region")["cell_type"].nunique().to_frame(name="count").reset_index()
    fig = _pie(count_type_df, "count", "region", cmap, height=350, width=350)
    fig.update_layout(showlegend=False)
    figs.append((fig, f"{dataset}_types_region_pie_all.pdf"))

    # Sign pies at full brain (drop NaN sign rows)
    sign_cmap = colors["sign"]
    m_sign = m_all[~m_all["sign"].isna()]
    count_df = m_sign.groupby("sign")["bodyId"].nunique().to_frame(name="count").reset_index()
    fig = _pie(count_df, "count", "sign", sign_cmap, height=350, width=350)
    fig.update_layout(showlegend=False)
    figs.append((fig, f"{dataset}_neuron_sign_pie.pdf"))

    count_type_df = m_sign.groupby("sign")["cell_type"].nunique().to_frame(name="count").reset_index()
    fig = _pie(count_type_df, "count", "sign", sign_cmap, height=350, width=350)
    fig.update_layout(showlegend=False)
    figs.append((fig, f"{dataset}_types_sign_pie_all.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_inventory_right_ol(
    meta: pd.DataFrame,
    colors: dict,
    *,
    dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """Pie charts of right-OL neurons / cell types by main_groups and sign."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "inventory"
    m = meta[(meta.region == "right OL") & (meta.main_groups != "nonOL")]

    figs = []
    for val in ["main_groups", "sign"]:
        cmap = colors[val]
        count_df = m.groupby(val)["bodyId"].nunique().to_frame(name="count").reset_index()
        fig = _pie(
            count_df, "count", val, cmap,
            title=f"distribution of {count_df['count'].sum()} OL neurons",
            height=350, width=350,
        )
        fig.update_layout(showlegend=False)
        figs.append((fig, f"{dataset}_OL_r_neuron_{val}_pie.pdf"))

        count_type_df = m.groupby(val)["cell_type"].nunique().to_frame(name="count").reset_index()
        fig = _pie(
            count_type_df, "count", val, cmap,
            title=f"distribution of {count_type_df['count'].sum()} OL instances",
            height=350, width=350,
        )
        fig.update_layout(showlegend=False)
        figs.append((fig, f"{dataset}_OL_r_types_{val}_pie_all.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_inventory_by_main_group(
    meta: pd.DataFrame,
    colors: dict,
    *,
    dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """For each main group in right OL, pie chart of sign distribution."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "inventory"
    m = meta[(meta.region == "right OL") & (meta.main_groups != "other") & (meta.instance != "R1-R6_R")]

    cmap = colors["sign"]
    figs = []
    for main_group in m["main_groups"].unique():
        plot_df = m[m["main_groups"] == main_group]

        count_df = plot_df.groupby("sign")["bodyId"].nunique().to_frame(name="count").reset_index()
        fig = _pie(
            count_df, "count", "sign", cmap,
            title=f"distribution of {count_df['count'].sum()} {main_group} neurons",
        )
        figs.append((fig, f"{dataset}_OL_r_neuron_{main_group}_sign_pie.pdf"))

        count_type_df = plot_df.groupby("sign")["cell_type"].nunique().to_frame(name="count").reset_index()
        fig = _pie(
            count_type_df, "count", "sign", cmap,
            title=f"distribution of {count_type_df['count'].sum()} {main_group} cell types",
        )
        figs.append((fig, f"{dataset}_OL_r_types_{main_group}_sign_pie_all.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %% [markdown]
# === 2. Flow ===


# %%
_FLOW_PLOT_TYPES = [
    "Mi1", "Mi9", "Tm5c", "T4a", "LPi2b", "C2",
    "Mi4", "TmY16", "TmY20", "Dm9", "Dm4", "Y13", "Cm11b",
    "MeTu1", "MeTu3c", "MeTu2a", "aMe4",
    "LPC2", "LPLC2", "LC4", "LC6", "LC9", "VS",
    "LPT111", "LC10a", "LPT23", "LoVP91",
]
_FLOW_STAR_TYPES = [
    "LC6", "LC4", "VS", "LPLC2", "Tm5c", "T4a",
    "C2", "Dm9", "TmY16", "aMe4", "Mi1",
]


# %%
def _plot_types(side_suffix="_R"):
    return [s + side_suffix for s in _FLOW_PLOT_TYPES], [s + side_suffix for s in _FLOW_STAR_TYPES]


# %%
def plot_coords(
    ol_meta: pd.DataFrame, *,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Hex heatmap of visual-input column counts."""
    from connectome_interpreter.external_map import hex_heatmap
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    df = ol_meta[ol_meta.main_groups == "visual input"]
    df = df.groupby("coords")["bodyId"].nunique().reset_index().rename(columns={"bodyId": "count"})
    df = df.set_index("coords")
    fig = hex_heatmap(
        df, custom_colorscale="RdBu_r", global_min=-6, global_max=6,
        dataset="mcns_right",
    )
    _save(fig, save_dir, f"{dataset}_coords.pdf")
    return fig


# %%
def plot_flow_sector_division(
    sector_map: pd.DataFrame, *,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Hex heatmap of the 5-sector retinotopic division."""
    from connectome_interpreter.external_map import hex_heatmap
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    fig = hex_heatmap(
        sector_map.set_index("coords")[["sector"]],
        custom_colorscale="RdBu_r", global_min=-5, global_max=-5,
        dataset="mcns_right",
    )
    _save(fig, save_dir, f"{dataset}_division.pdf")
    return fig


# %%
def plot_flow_by_main_groups(
    flow_type_ol: pd.DataFrame, conn_ol: pd.DataFrame,
    type_dir_ol: pd.DataFrame, flow_ol: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame,
    *, dataset: str = DATASET, hit_diff_thre: float = 0.5, save_dir: Path | None = None,
):
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    figs = []

    # fraction of cell types per layer by main_groups
    bins = np.arange(-0.5, 6 + 0.5, 1)
    fig = _hist_lines(flow_type_ol, "hitting_time", "main_groups", bins, colored_main_groups_df)
    fig.update_layout(
        xaxis_title="layer", xaxis_range=[-.1, 5.1], xaxis_tickvals=[0, 1, 2, 3, 4, 5],
        yaxis_title="frac. cell types", yaxis_range=[-.01, 1.01],
    )
    figs.append((fig, f"{dataset}_0.1thre_hit_hist_frac_groups_types.pdf"))

    # fraction of weights vs layer difference by main_groups_pre
    diff_bins = np.array(range(-4, 4)).astype(float) + .5
    fig = _hist_sum_lines(conn_ol, "hitting_time_diff", "weight", "main_groups_pre",
                          diff_bins, colored_main_groups_df)
    fig.update_layout(
        xaxis_title="post - pre layer", yaxis_title="frac. weights",
        xaxis_range=[-3.2, 3.2], xaxis_tickvals=[-3, -2, -1, 0, 1, 2, 3],
        yaxis_range=[-.01, 1.01],
    )
    figs.append((fig, f"{dataset}_0.1thre_diff_hit_hist_frac_groups_conn.pdf"))

    # type-level ff/fb/la distributions by main_groups_pre
    ff_bins = np.arange(-0.1, 1.2, 0.2)
    for direct in ["ff", "fb", "la"]:
        fig = _hist_lines(type_dir_ol, f"frac_{direct}", "main_groups_pre",
                          ff_bins, colored_main_groups_df)
        fig.update_layout(
            xaxis_title=f"frac_{direct}", xaxis_range=[-.05, 1.05],
            yaxis_title="frac. cell types", yaxis_range=[-.01, 1.01],
        )
        figs.append((fig, f"{dataset}_{direct}_frac_hist_all_groups.pdf"))

    # fine-grained layer histogram (raw counts)
    fine_bins = np.arange(-0.25 / 2, 6 + 0.25 / 2, .5 / 2)
    fig = _hist_lines(flow_ol, "hitting_time_median", "main_groups",
                      fine_bins, colored_main_groups_df, norm=False)
    for i in range(6):
        fig.add_vline(x=i, line_width=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="layer", xaxis_range=[-.1, 3.3], xaxis_tickvals=[0, 1, 2, 3, 4, 5],
        yaxis_title="no. neurons",
    )
    figs.append((fig, f"{dataset}_0.1thre_hit_hist_fine_tot_groups_types.pdf"))

    # fine-grained diff histogram (sum weights, raw)
    diff_fine = np.arange(-4 - 0.25 / 2, 4 + 0.25 / 2, .5 / 2)
    fig = _hist_sum_lines(conn_ol, "hitting_time_diff", "weight", "main_groups_pre",
                          diff_fine, colored_main_groups_df, norm=False)
    for i in range(7):
        fig.add_vline(x=-3 + i, line_width=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="post - pre layer", yaxis_title="no. synapses",
        xaxis_range=[-1.6, 1.6], xaxis_tickvals=[-3, -2, -1, 0, 1, 2, 3],
    )
    figs.append((fig, f"{dataset}_0.1thre_diff_hit_hist_fine_tot_groups_conn.pdf"))

    # fine-grained diff histogram (pair counts)
    conn_pairs = conn_ol.copy()
    conn_pairs["weight"] = 1
    fig = _hist_sum_lines(conn_pairs, "hitting_time_diff", "weight", "main_groups_pre",
                          diff_fine, colored_main_groups_df, norm=False)
    for i in range(7):
        fig.add_vline(x=-3 + i, line_width=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="post - pre layer", yaxis_title="no. partners",
        xaxis_range=[-1.6, 1.6], xaxis_tickvals=[-3, -2, -1, 0, 1, 2, 3],
    )
    figs.append((fig, f"{dataset}_0.1thre_diff_hit_hist_fine_tot_groups_pairs.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_flow_by_sign(
    flow_type_ol: pd.DataFrame, conn_ol: pd.DataFrame,
    type_dir_ol: pd.DataFrame, flow_ol: pd.DataFrame,
    colored_sign_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    figs = []

    bins = np.arange(-0.5, 6 + 0.5, 1)
    fig = _hist_lines(flow_type_ol, "hitting_time", "sign", bins, colored_sign_df)
    fig.update_layout(
        xaxis_title="layer", xaxis_range=[-.1, 5.1],
        yaxis_title="frac. cell types", yaxis_range=[-.01, 1.01],
    )
    figs.append((fig, f"{dataset}_0.1thre_hit_hist_frac_signs_types.pdf"))

    diff_bins = np.array(range(-4, 4)).astype(float) + .5
    fig = _hist_sum_lines(conn_ol, "hitting_time_diff", "weight", "sign_pre",
                          diff_bins, colored_sign_df)
    fig.update_layout(
        xaxis_title="post - pre layer", yaxis_title="frac. weights",
        xaxis_range=[-3.2, 3.2], xaxis_tickvals=[-3, -2, -1, 0, 1, 2, 3],
        yaxis_range=[-.01, 1.01],
    )
    figs.append((fig, f"{dataset}_0.1thre_diff_hit_hist_frac_signs_conn.pdf"))

    ff_bins = np.arange(-0.1, 1.2, 0.2)
    for direct in ["ff", "fb", "la"]:
        fig = _hist_lines(type_dir_ol, f"frac_{direct}", "sign_pre",
                          ff_bins, colored_sign_df, width=450)
        fig.update_layout(
            xaxis_title=f"frac_{direct}", xaxis_range=[-.05, 1.05],
            yaxis_title="frac. cell types", yaxis_range=[-.01, 1.01],
        )
        figs.append((fig, f"{dataset}_{direct}_frac_hist_all_signs.pdf"))

    fine_bins = np.arange(-0.25 / 2, 6 + 0.25 / 2, .5 / 2)
    fig = _hist_lines(flow_ol[flow_ol.sign != 0], "hitting_time_median", "sign",
                      fine_bins, colored_sign_df, norm=False, width=450)
    for i in range(6):
        fig.add_vline(x=i, line_width=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="layer", xaxis_range=[-.1, 3.3], xaxis_tickvals=[0, 1, 2, 3, 4, 5],
        yaxis_title="no. neurons", yaxis_range=[-1.5e3, 19e3],
    )
    figs.append((fig, f"{dataset}_0.1thre_hit_hist_fine_tot_signs_types.pdf"))

    diff_fine = np.arange(-4 - 0.25 / 2, 4 + 0.25 / 2, .5 / 2)
    fig = _hist_sum_lines(conn_ol[conn_ol.sign_pre != 0], "hitting_time_diff", "weight",
                          "sign_pre", diff_fine, colored_sign_df, norm=False, width=450)
    for i in range(7):
        fig.add_vline(x=-3 + i, line_width=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="post - pre layer", yaxis_title="no. synapses",
        xaxis_range=[-1.6, 1.6], xaxis_tickvals=[-3, -2, -1, 0, 1, 2, 3],
    )
    figs.append((fig, f"{dataset}_0.1thre_diff_hit_hist_fine_tot_signs_conn.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_flow_type_scatter_and_triangle(
    flow_type_ol: pd.DataFrame, type_dir_ol: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    figs = []
    _, star_types = _plot_types()

    fig = _scatter(flow_type_ol, "count", "hitting_time", "main_groups", "instance",
                   colored_main_groups_df, star_types)
    fig.update_layout(xaxis_title="no. neurons", yaxis_title="layer")
    fig.update_xaxes(type="log")
    figs.append((fig, f"{dataset}_examples_0.1thre_no_vs_layer.pdf"))

    eps = 5e-2
    d = type_dir_ol.copy()
    d[["frac_la", "frac_fb", "frac_ff"]] = d[["frac_la", "frac_fb", "frac_ff"]] * (1 - eps) + eps / 3
    fig = _directedness_triangle(d, "main_groups_pre", "instance_pre",
                                 colored_main_groups_df["color"].to_dict(), star_types)
    figs.append((fig, f"{dataset}_examples_0.1thre_dir_triangle.pdf"))

    d["sign_pre"] = d["sign_pre"].astype(str)
    cmap_sign = colored_sign_df.copy()
    cmap_sign.index = cmap_sign.index.astype(str)
    fig = _directedness_triangle(d, "sign_pre", "instance_pre",
                                 cmap_sign["color"].to_dict(), star_types)
    figs.append((fig, f"{dataset}_examples_0.1thre_dir_triangle_sign.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_flow_type_boxes(
    flow_df: pd.DataFrame, dir_ol: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    figs = []
    plot_types, _ = _plot_types()

    flow_sub = flow_df[flow_df["instance"].isin(plot_types)].copy()
    flow_sub["cell_type"] = [s[:-2] for s in flow_sub["instance"]]
    color_map = dict(zip(
        flow_sub["cell_type"],
        colored_main_groups_df.loc[flow_sub["main_groups"], "color"].values,
    ))

    grouped = flow_sub.groupby(["instance", "cell_type"])["hitting_time"].median().reset_index()
    sorted_types = grouped.loc[grouped.sort_values(["hitting_time", "instance"]).index, "cell_type"].values
    flow_sub["cell_type"] = pd.Categorical(flow_sub["cell_type"], categories=sorted_types, ordered=True)
    flow_sub = flow_sub.sort_values("cell_type")

    dir_sub = dir_ol[dir_ol["instance_pre"].isin(plot_types)].copy()
    dir_sub["cell_type_pre"] = pd.Categorical(dir_sub["cell_type_pre"], categories=sorted_types, ordered=True)
    dir_sub = dir_sub.sort_values("cell_type_pre")

    # Pad cells with no outgoing connections as zero rows so frac medians
    # cover the full roster per type (matches plot_flow_numerous_type_boxes).
    type_totals = (
        flow_sub.groupby("cell_type", observed=True).size()
        .reindex(sorted_types, fill_value=0)
    )
    type_present = (
        dir_sub.groupby("cell_type_pre", observed=True).size()
        .reindex(sorted_types, fill_value=0)
    )
    n_missing = (type_totals - type_present).clip(lower=0).astype(int)
    pad_rows = pd.DataFrame({
        "cell_type_pre": pd.Categorical(
            np.repeat(sorted_types, n_missing.values),
            categories=sorted_types, ordered=True,
        ),
        "frac_ff": 0.0, "frac_fb": 0.0, "frac_la": 0.0,
    })
    dir_sub = pd.concat([dir_sub, pad_rows], ignore_index=True).sort_values("cell_type_pre")

    fig = _box(flow_sub, "hitting_time", "cell_type", "cell_type", color_map, height=650)
    for i in range(4):
        fig.add_vline(x=i + 1, line_width=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title="layer", yaxis_title="")
    fig.update_traces(jitter=0.4, marker=dict(size=3, opacity=0.4), selector=dict(type="box"))
    figs.append((fig, f"{dataset}_examples_0.1thre_hit_hist.pdf"))

    for direct in ["ff", "fb", "la"]:
        fig = _box(dir_sub, f"frac_{direct}", "cell_type_pre", "cell_type_pre",
                   color_map, width=250, height=650)
        fig.update_layout(xaxis_range=[-0.05, 1.05])
        fig.update_traces(jitter=0.4, marker=dict(size=3, opacity=0.4), selector=dict(type="box"))
        figs.append((fig, f"{dataset}_examples_0.1thre_{direct}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_flow_numerous_type_boxes(
    flow_ol: pd.DataFrame, dir_ol: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    groups: list[tuple[int, int]] = ((100, 2000), (28, 100)),
    height: int = 3000,
):
    """Box plots across **all** numerous cell types (legacy nb 2 `numerous{1,2}`
    cells). For each `(n_min, n_max)` group: hitting-time box + frac_ff/fb/la
    boxes with a black marker per type indicating its dominant direction.
    Default groups: (100, 2000] and (28, 100] per neuron count.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    figs = []
    flow_sub_base = flow_ol[flow_ol["main_groups"] != "other"]

    for j, (n_min, n_max) in enumerate(groups):
        numerous = flow_sub_base[
            (flow_sub_base["count"] > n_min) & (flow_sub_base["count"] <= n_max)
        ]["instance"].unique()
        if len(numerous) == 0:
            continue
        flow_sub = flow_sub_base[flow_sub_base["instance"].isin(numerous)].copy()
        flow_sub["cell_type"] = [s[:-2] for s in flow_sub["instance"]]
        color_map = dict(zip(
            flow_sub["cell_type"],
            colored_main_groups_df.loc[flow_sub["main_groups"], "color"].values,
        ))
        grouped = flow_sub.groupby("cell_type")["hitting_time"].median().reset_index()
        sorted_types = grouped.loc[grouped.sort_values("hitting_time").index, "cell_type"].values
        flow_sub["cell_type"] = pd.Categorical(
            flow_sub["cell_type"], categories=sorted_types, ordered=True,
        )
        flow_sub = flow_sub.sort_values("cell_type")

        dir_sub = dir_ol[dir_ol["instance_pre"].isin(numerous)].copy()
        dir_sub["cell_type_pre"] = pd.Categorical(
            dir_sub["cell_type_pre"], categories=sorted_types, ordered=True,
        )
        dir_sub = dir_sub.sort_values("cell_type_pre")

        fig = _box(flow_sub, "hitting_time", "cell_type", "cell_type",
                   color_map, height=height)
        for i in range(4):
            fig.add_vline(x=i + 1, line_width=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(xaxis_title="layer", xaxis_range=[-0.1, 3.3])
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(sorted_types))),
            ticktext=list(sorted_types)[::-1],
            range=[-0.5, len(sorted_types) - 0.5],
        )
        fig.update_traces(jitter=0.4, marker=dict(size=3, opacity=0.4),
                          selector=dict(type="box"))
        figs.append((fig, f"{dataset}_numerous{j + 1}_0.1thre_hit_hist.pdf"))

        # Cells of a type with no outgoing connections are absent from dir_sub;
        # pad them as zero rows so the median is over the full roster per type.
        type_totals = (
            flow_sub.groupby("cell_type", observed=True).size()
            .reindex(sorted_types, fill_value=0)
        )
        type_present = (
            dir_sub.groupby("cell_type_pre", observed=True).size()
            .reindex(sorted_types, fill_value=0)
        )
        n_missing = (type_totals - type_present).clip(lower=0).astype(int)
        pad_rows = pd.DataFrame({
            "cell_type_pre": pd.Categorical(
                np.repeat(sorted_types, n_missing.values),
                categories=sorted_types, ordered=True,
            ),
            "frac_ff": 0.0, "frac_fb": 0.0, "frac_la": 0.0,
        })
        dir_sub = pd.concat([dir_sub, pad_rows], ignore_index=True).sort_values("cell_type_pre")
        dir_medians = dir_sub.groupby("cell_type_pre", observed=True)[
            ["frac_ff", "frac_fb", "frac_la"]
        ].median()
        frac_max = dir_medians.idxmax(axis=1).values
        for direct in ["ff", "fb", "la"]:
            fig = _box(dir_sub, f"frac_{direct}", "cell_type_pre", "cell_type_pre",
                       color_map, width=350, height=height)
            idx_sub = np.where(frac_max == f"frac_{direct}")[0]
            yvals = pd.Categorical(
                dir_medians.index[idx_sub],
                categories=sorted_types, ordered=True,
            )
            fig.add_trace(go.Scatter(
                x=dir_medians.iloc[idx_sub][f"frac_{direct}"].values,
                y=yvals, mode="markers",
                marker=dict(color="black", size=10),
                showlegend=False,
            ))
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(len(sorted_types))),
                ticktext=list(sorted_types)[::-1],
                range=[-0.5, len(sorted_types) - 0.5],
            )
            fig.update_layout(xaxis_range=[-0.05, 1.05])
            fig.update_traces(jitter=0.4, marker=dict(size=3, opacity=0.4),
                              selector=dict(type="box"))
            figs.append((fig, f"{dataset}_numerous{j + 1}_0.1thre_{direct}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_flow_connectivity_heatmap(
    conn_ol: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    figs = []

    bins = np.arange(-0.5 / 2, 6 + 0.5, 1 / 2)
    centers = (bins[1:] + bins[:-1]) / 2
    n_plot = int((bins <= 4).sum())
    tick_labels = ["%.1f" % i for i in centers[:n_plot][::1]]

    cb = conn_ol.copy()
    cb["bin_pre"] = pd.cut(cb["hitting_time_pre"], bins=bins, labels=centers).astype(float)
    cb["bin_post"] = pd.cut(cb["hitting_time_post"], bins=bins, labels=centers).astype(float)
    full = cb.groupby(["bin_pre", "bin_post"], observed=False)["weight"].sum().unstack().fillna(0)
    mat = full.reindex(index=centers, columns=centers, fill_value=0)

    fig = _heatmatrix(mat, tick_labels, cmap="Greys")
    fig.update_yaxes(autorange="reversed")
    figs.append((fig, f"{dataset}_0.1thre_conn_matrix.pdf"))

    for group in colored_main_groups_df.index:
        sub = cb[cb["main_groups_pre"] == group]
        m = sub.groupby(["bin_pre", "bin_post"], observed=False)["weight"].sum().unstack().fillna(0)
        m = m.reindex(index=centers, columns=centers, fill_value=0)
        color = colored_main_groups_df.loc[group, "color"]
        cs = [[0.0, "white"], [1.0, color]]
        fig = _heatmatrix(m, tick_labels, cmap=cs)
        fig.update_layout(title=group)
        fig.update_yaxes(autorange="reversed")
        figs.append((fig, f"{dataset}_0.1thre_conn_matrix_{group}.pdf"))

    for sign in colored_sign_df.index.astype("int"):
        sub = cb[cb["sign_pre"] == sign]
        m = sub.groupby(["bin_pre", "bin_post"], observed=False)["weight"].sum().unstack().fillna(0)
        m = m.reindex(index=centers, columns=centers, fill_value=0)
        color = colored_sign_df.loc[sign, "color"]
        cs = [[0.0, "white"], [1.0, color]]
        fig = _heatmatrix(m, tick_labels, cmap=cs)
        fig.update_layout(title=sign)
        fig.update_yaxes(autorange="reversed")
        figs.append((fig, f"{dataset}_0.1thre_conn_matrix_{sign}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_flow_threshold_choice(
    conn_ol: pd.DataFrame,
    *, dataset: str = DATASET, hit_diff_thre: float = 0.5, save_dir: Path | None = None,
):
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    thresholds = np.linspace(-1, 1, 50)
    total_w = conn_ol["weight"].sum()
    fracs = np.array([
        conn_ol[conn_ol["hitting_time_diff"] >= thr]["weight"].sum() for thr in thresholds
    ]) / total_w
    y0 = fracs[np.where(thresholds > 0)[0][0]]
    plot_df = pd.DataFrame({"thre": thresholds, "frac": fracs})
    fig = _point_line(plot_df, "thre", "frac", x_thre=0, y_thre=y0)
    fig.update_layout(
        xaxis_title="post - pre layer threshold",
        yaxis_title="frac. weights above layer threshold",
        xaxis_range=[-1.05, 1.05], yaxis_range=[-0.05, 1.05],
    )
    _save(fig, save_dir, f"{dataset}_threshold_fraction_connections.pdf")
    return fig


# %%
def plot_ol_layers_violin(
    layers: pd.DataFrame, rois: list[str], labels: list[str],
    *, dataset: str = DATASET, filename_stem: str,
    save_dir: Path | None = None, height: int = 800, width: int = 500,
):
    """Horizontal violin of `hitting_time` across the given OL ROIs.

    `rois` is the ordered list of ROI names to display (top → bottom); `labels`
    are the short display names (e.g. `M1..M10, AME`). `filename_stem` goes
    into the output filename as `{dataset}_{filename_stem}.png`."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    sub = layers[layers["roi"].isin(rois)].copy()
    sub["roi"] = pd.Categorical(sub["roi"], categories=rois, ordered=True)
    sub = sub.sort_values("roi")
    fig = _violin_median(
        sub, "hitting_time", "roi", labels=labels, height=height, width=width,
        xaxis_title="layer", xaxis_tickvals=[0, 1, 2, 3, 4],
        xaxis_range=[-0.2, 4.1],
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.write_image(save_dir / f"{dataset}_{filename_stem}.png", scale=2)
    return fig


# %% [markdown]
# === 3. Clusters ===


# %%
_CLUSTER_BLUES = ["#041f4a", "#08306b", "#08519c", "#2171b5", "#4292c6",
                  "#6baed6", "#9ecae1", "#c6dbef", "#deebf7"]
_CLUSTER_TYPES = [
    "l-LNv", "MeTu2a", "aMe5", "MeTu1", "MeTu3c", "LC6", "MeTu4f", "LPLC2", "LC4",
]
_OUT_TYPES = [
    "aMe3", "LPLC2", "LC4", "LC6",
    "MeTu3a", "MeTu3b", "MeTu3c", "VS",
    "MeTu2a", "LPLC1", "LPLC2",
    "MeTu1", "LC10a", "aMe12", "MeVP11",
]
_PART_TYPES = [
    "L1", "L2", "L3", "R7", "R8", "R7d", "R8d", "HBeyelet",
    "T4a", "Dm4", "Dm9", "Tm31", "T5a",
    "Tm5", "Tm20", "Mi1", "Mi4", "Dm12",
    "Mi15", "TmY18", "Dm3a", "L4", "L5",
    "Y13", "Cm3", "Cm15",
    "aMe6b", "aMe6c", "Dm-DRA1", "Dm-DRA2", "Dm20", "Cm14",
    "Cm11d", "Mi9", "Tm1", "Tm2", "Tm3", "Tm9", "Tm5a",
    "TmY16", "Li25", "CT1", "Dm8", "TmY21", "TmY4", "Dm2", "Mi10",
    "Pm13", "Tm33", "Y11", "T1", "C2", "C3", "MeLo1",
    "Dm8a", "Dm8b", "Dm9", "Tlp1", "Am1", "LPi14", "Li38", "T2", "T2a", "T3",
]


# %%
def _cluster_color_df(n_clu: int) -> pd.DataFrame:
    names = [f"oc{i + 1}" for i in range(n_clu)]
    blues = _CLUSTER_BLUES[:n_clu] if n_clu <= len(_CLUSTER_BLUES) else (
        _CLUSTER_BLUES * ((n_clu // len(_CLUSTER_BLUES)) + 1)
    )[:n_clu]
    return pd.DataFrame(blues, index=names, columns=["color"])


# %%
def plot_clusters_dendrograms(
    feature_df: pd.DataFrame, frac_thre: float = 0.21, *,
    method: str = "ward", metric: str = "euclidean",
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
    compact_lastp: int = 10,
):
    """Three dendrogram PDFs (lastp=50, compact lastp, full vertical)."""
    import matplotlib.pyplot as plt
    import scipy.cluster.hierarchy as sch
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    save_dir.mkdir(parents=True, exist_ok=True)

    feature_vec = feature_df.values / feature_df.values.sum(1)[:, np.newaxis]
    Z = sch.linkage(feature_vec, method=method, metric=metric)
    thre = frac_thre * Z[:, 2].max()

    plt.figure(figsize=(7.5, 6))
    sch.dendrogram(Z, p=50, truncate_mode="lastp", above_threshold_color="black",
                   link_color_func=lambda k: "black")
    plt.xticks([])
    plt.savefig(save_dir / f"{dataset}{side_str}_dendrogram_clusters_50.pdf",
                dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 6))
    sch.dendrogram(Z, p=compact_lastp, truncate_mode="lastp", above_threshold_color="black",
                   link_color_func=lambda k: "black")
    plt.xticks([])
    plt.savefig(save_dir / f"{dataset}{side_str}_dendrogram_clusters.pdf",
                dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(3, 65))
    dendr = sch.dendrogram(
        Z, color_threshold=thre, show_leaf_counts=True,
        labels=[s[:-2] for s in feature_df.index.values], leaf_font_size=12,
        above_threshold_color="black", orientation="right",
    )
    plt.axvline(x=thre, color="gray", linestyle="--")
    plt.ylabel("clustered OL output types", fontsize=14)
    plt.xticks(
        ticks=[0, thre, 0.5 * Z[:, 2].max(), Z[:, 2].max()],
        labels=["0", f"{frac_thre:.2f}", "0.5", "1"],
    )
    plt.xlabel("linkage distance/max", fontsize=14)
    plt.savefig(save_dir / f"{dataset}{side_str}_dendrogram_full.pdf",
                dpi=300, bbox_inches="tight")
    plt.close()

    return Z, thre, dendr


# %%
def plot_clusters_correlation_matrix(
    feature_df: pd.DataFrame, dendr: dict, cluster_num: pd.Series, *,
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
):
    """Heatmap of cosine-like correlation between feature vectors with cluster boundaries."""
    from scipy.stats import zscore
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    n_features = feature_df.shape[1]
    feature_vec = zscore(feature_df.values, axis=1)
    corr = (feature_vec @ feature_vec.T) / n_features
    corr[np.isnan(corr)] = 0
    order = dendr["leaves"][::-1]
    fig = _heatmatrix(corr[order][:, order], feature_df.index.values,
                      cmap="RdBu_r", height=600, width=600)
    n_clu = len(cluster_num)
    cumulative = np.cumsum(cluster_num.values)
    for i in range(n_clu):
        fig.add_hline(y=cumulative[i] - .5, line_width=.5, line_color="gray")
        fig.add_vline(x=cumulative[i] - .5, line_width=.5, line_color="gray")
    fig.update_layout(
        xaxis_title="clustered OL output types", xaxis_tickvals=[],
        yaxis_title="clustered OL output types", yaxis_tickvals=[],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(zmin=-1, zmax=1)
    _save(fig, save_dir, f"{dataset}{side_str}_clustering_matrix.pdf")
    return fig


# %%
def plot_clusters_example_features(
    feature_df: pd.DataFrame, in_instances, cluster_types,
    colored_seed_df: pd.DataFrame, *,
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
):
    """Stacked bar of normalised feature vector for example output types."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    sub = feature_df.loc[cluster_types, in_instances]
    plot_df = sub / sub.sum(1).values[:, np.newaxis]
    plot_df.index = [s[:-2] for s in plot_df.index]
    plot_df.columns = [s[:-2] for s in plot_df.columns]
    plot_df = plot_df.stack(0).reset_index(name="effective weight").rename(
        columns={"level_0": "out type", "level_1": "in type"}
    )
    fig = _stacked_bars(plot_df, "out type", "effective weight", "in type",
                        colored_seed_df["color"].to_dict(),
                        xrot=-45, height=400, width=550)
    _save(fig, save_dir, f"{dataset}{side_str}_in_out_examples.pdf")
    return fig


# %%
def plot_clusters_pathway_features(
    feature_df: pd.DataFrame, in_instances, cluster_df: pd.DataFrame,
    cluster_num: pd.Series, colored_seed_df: pd.DataFrame, *,
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
):
    """Stacked bar of cluster-mean normalised feature vector."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    sub = feature_df.loc[:, in_instances] / feature_df.loc[:, in_instances].sum(1).values[:, np.newaxis]
    df = sub.reset_index(names="instance").merge(cluster_df, on="instance")
    df = df.groupby("cluster")[in_instances].mean()
    df.columns = [s[:-2] for s in df.columns]
    df = df.stack(0).reset_index(name="mean effective weight").rename(
        columns={"level_0": "cluster", "level_1": "in type"}
    )
    fig = _stacked_bars(df, "cluster", "mean effective weight", "in type",
                        colored_seed_df["color"].to_dict(), height=400, width=550)
    n_clu = len(cluster_num)
    fig.update_layout(
        xaxis_tickvals=np.arange(1, n_clu + 1),
        xaxis_ticktext=[f"oc{i}<br>({cluster_num[i]})" for i in range(1, n_clu + 1)],
    )
    _save(fig, save_dir, f"{dataset}{side_str}_in_out_pathways.pdf")
    return fig


# %%
def plot_clusters_scatter(
    cluster_df: pd.DataFrame, flow_type_ol: pd.DataFrame, type_dir_ol: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None, jitter: float = 0.2,
):
    """Scatter of layer / ff-fb-la fractions vs cluster, coloured by cluster and by sign."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    figs = []
    n_clu = int(cluster_df["cluster"].max())
    cluster_color = _cluster_color_df(n_clu)
    cluster_names = list(cluster_color.index)
    out_types_r = list(set([s + "_R" for s in _OUT_TYPES]) | set([s + "_R" for s in _CLUSTER_TYPES]))

    base = cluster_df.merge(flow_type_ol, how="left", on="instance").copy()
    base["cluster_name"] = ["oc" + str(i) for i in base["cluster"]]
    rng = np.random.default_rng(0)
    base["y"] = base["cluster"].astype(float) + rng.uniform(-jitter, jitter, size=len(base))

    layer_df = base[~np.isnan(base.hitting_time)]
    fig = _scatter(layer_df, "hitting_time", "y", "cluster_name", "instance",
                   cluster_color, out_types_r, height=400, width=500)
    fig.update_layout(yaxis_title="cluster", xaxis_title="layer")
    fig.update_yaxes(autorange="reversed", tickvals=np.arange(1, n_clu + 1), ticktext=cluster_names)
    figs.append((fig, f"{dataset}_examples_layer_vs_cluster.pdf"))

    fig = _scatter(layer_df, "hitting_time", "y", "sign", "instance",
                   colored_sign_df, out_types_r, height=400, width=500)
    fig.update_layout(yaxis_title="cluster", xaxis_title="layer")
    fig.update_yaxes(autorange="reversed", tickvals=np.arange(1, n_clu + 1), ticktext=cluster_names)
    figs.append((fig, f"{dataset}_examples_layer_vs_cluster_sign.pdf"))

    dir_df = type_dir_ol.rename(
        columns={"instance_pre": "instance", "main_groups_pre": "main_groups", "sign_pre": "sign"}
    ).merge(cluster_df, how="right", on="instance")
    dir_df["cluster_name"] = ["oc" + str(i) for i in dir_df["cluster"]]
    dir_df["y"] = dir_df["cluster"].astype(float) + rng.uniform(-jitter, jitter, size=len(dir_df))
    dir_df = dir_df[~np.isnan(dir_df.frac_ff)]

    for direct in ["ff", "fb", "la"]:
        for c_col, cmap in [("cluster_name", cluster_color), ("sign", colored_sign_df)]:
            fig = _scatter(dir_df, f"frac_{direct}", "y", c_col, "instance",
                           cmap, out_types_r, height=400, width=500)
            fig.update_layout(yaxis_title="cluster", xaxis_title=f"frac_{direct}")
            fig.update_yaxes(autorange="reversed", tickvals=np.arange(1, n_clu + 1), ticktext=cluster_names)
            suffix = "" if c_col == "cluster_name" else "_sign"
            figs.append((fig, f"{dataset}_examples_{direct}_vs_cluster{suffix}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_clusters_participation(
    type_part_df: pd.DataFrame, ol_meta: pd.DataFrame, ol_flow_type: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None, jitter: float = 0.2,
):
    """Participation: n_clusters histogram, max-pi vs layer / vs idxmax, and per-cluster pi."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    figs = []
    cluster_names = list(type_part_df.columns[type_part_df.columns.str.contains(r"^c\d+$")].values)
    n_clu = len(cluster_names)

    type_part = (
        type_part_df
        .merge(ol_meta[["instance", "main_groups", "sign"]].drop_duplicates(),
               on="instance", how="left")
        .merge(ol_flow_type[["instance", "hitting_time"]].drop_duplicates(),
               on="instance", how="left")
    )

    bins = np.linspace(0, n_clu + 1, n_clu + 2) - 0.5
    color_val = {"main_groups": colored_main_groups_df, "sign": colored_sign_df}
    base = type_part[type_part["main_groups"].isin(["visual input", "OL internal"])]
    for val in ["main_groups", "sign"]:
        plot_df = base if val != "sign" else base[base.sign != 0]
        fig = _hist_lines(plot_df, "n_clusters", val, bins, color_val[val])
        fig.update_layout(
            xaxis_title=f"no. clusters with pi>1/{n_clu}",
            xaxis_range=[-.1, n_clu + 0.1], xaxis_tickvals=np.arange(n_clu + 1),
            yaxis_title="frac. cell types", yaxis_range=[-.01, 0.66],
        )
        figs.append((fig, f"{dataset}_in_out_n_clu_participation_{val}.pdf"))

    part_types_r = [s + "_R" for s in _PART_TYPES]
    for val in ["main_groups", "sign"]:
        fig = _scatter(base, "hitting_time", "max", val, "instance",
                       color_val[val], part_types_r, height=400, width=500)
        fig.add_hline(y=1 / n_clu, line_dash="dot", line_color="gray")
        fig.update_layout(yaxis_title="max. pi", xaxis_title="layer", yaxis_range=[-0.05, 1.05])
        figs.append((fig, f"{dataset}_examples_layer_max_participation_{val}.pdf"))

    base2 = base.reset_index(drop=True).copy()
    cat_to_num = {cat[1:]: i for i, cat in enumerate(cluster_names)}
    rng = np.random.default_rng(0)
    base2["x"] = base2["idxmax"].map(cat_to_num).astype(float) + rng.uniform(-jitter, jitter, size=len(base2))
    for val in ["main_groups", "sign"]:
        fig = _scatter(base2, "x", "max", val, "instance",
                       color_val[val], part_types_r, height=400, width=500)
        fig.add_hline(y=1 / n_clu, line_dash="dot", line_color="gray")
        fig.update_layout(
            yaxis_title="max. pi", xaxis_title="max. ci", xaxis_tickangle=0,
            xaxis_tickvals=np.arange(n_clu),
            xaxis_ticktext=["o" + c for c in cluster_names],
            yaxis_range=[-0.05, 1.05],
        )
        figs.append((fig, f"{dataset}_examples_max_participation_{val}.pdf"))

    long_df = type_part.melt(
        id_vars=["instance", "main_groups", "sign", "hitting_time"],
        value_vars=cluster_names, var_name="cluster", value_name="participation",
    )
    long_df = long_df[long_df["participation"] >= 2 / n_clu]
    long_df = long_df[long_df["main_groups"].isin(["visual input", "OL internal"])].reset_index(drop=True)
    x_num = np.array([int(c[1:]) for c in long_df["cluster"].values])
    long_df["x"] = x_num + rng.uniform(-jitter, jitter, size=len(long_df))
    for val in ["main_groups", "sign"]:
        fig = _scatter(long_df, "x", "participation", val, "instance",
                       color_val[val], part_types_r, height=400, width=500)
        fig.add_hline(y=2 / n_clu, line_dash="dot", line_color="gray")
        fig.update_layout(
            yaxis_title="pi", xaxis_title="ci", xaxis_tickangle=0,
            xaxis_tickvals=np.arange(n_clu) + 1,
            xaxis_ticktext=["o" + c for c in cluster_names],
            yaxis_range=[-0.05, 1.05],
        )
        figs.append((fig, f"{dataset}_examples_participation_{val}.pdf"))

    for fig, name in figs:
        # main_2_h panels (n_clu participation histograms) are static — no HTML
        _save(fig, save_dir, name, html="in_out_n_clu_participation" not in name)
    return figs


# %%
def plot_clusters_intra_connectivity(
    intra: dict, *, dataset: str = DATASET, save_dir: Path | None = None,
    filename_stem: str | None = None,
):
    """Cluster-cluster connectivity heatmap (output-normalised).

    Filename defaults to `{dataset}_outrel_{n_clu}_conn_matrix_clusters.pdf`
    (n_clu inferred from `intra['n_clu']`); pass an explicit `filename_stem`
    to override."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    n = intra["n_clu"]
    tick_labels = [f"c{i}" for i in range(1, n + 1)]
    fig = _heatmatrix(intra["matrix"], tick_labels, cmap="Greys", height=600, width=600)
    fig.update_layout(
        title=f"all weights: {intra['total_weight']: .0f}",
        xaxis_title="post", yaxis_title="pre",
    )
    fig.update_layout(xaxis_tickangle=0)
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(colorbar=dict(tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"]),
                      zmin=0, zmax=1)
    if filename_stem is None:
        filename_stem = f"outrel_{n}_conn_matrix_clusters"
    _save(fig, save_dir, f"{dataset}_{filename_stem}.pdf")
    return fig


# %%
def plot_clusters_pathway_connectivity(
    pathways: dict, n_clu: int, *,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Pathway-level connectivity heatmaps (cluster ↔ cluster, with/without self;
    plus the 5 e/i breakdowns with/without self)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    figs = []
    cluster_names = [f"c{i}" for i in range(1, n_clu + 1)]
    title_list = ["all", "ee", "ei", "ie", "ii"]
    cmap_list = ["Greys", "Reds", "Reds", "Blues", "Blues"]

    conn0 = pathways["conn"][0]
    diag0 = pathways["diag"][0]

    # cluster-cluster (drop in/out endpoints)
    inner = conn0[1:-1, 1:-1]
    tot = inner.sum()
    plot_mat = inner / inner.sum(1)[:, np.newaxis]
    fig = _heatmatrix(plot_mat, cluster_names, cmap="Greys", height=600, width=600)
    fig.update_layout(title=f"all weights: {tot: .0f}", xaxis_title="post", yaxis_title="pre")
    fig.update_layout(xaxis_tickangle=0)
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(colorbar=dict(tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"]),
                      zmin=0, zmax=1)
    figs.append((fig, f"{dataset}_outrel_conn_matrix_clusters.pdf"))

    inner_no_self = conn0[1:-1, 1:-1] - np.diag(diag0)[1:-1, 1:-1]
    tot = inner_no_self.sum()
    plot_mat = inner_no_self / inner_no_self.sum(1)[:, np.newaxis]
    fig = _heatmatrix(plot_mat, cluster_names, cmap="Greys", height=600, width=600)
    fig.update_layout(title=f"without cell type-intrinsic weights: {tot: .0f}",
                      xaxis_title="post", yaxis_title="pre")
    fig.update_layout(xaxis_tickangle=0)
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(colorbar=dict(tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"]),
                      zmin=0, zmax=1)
    figs.append((fig, f"{dataset}_outrel_conn_matrix_clusters_wo_selfconn.pdf"))

    # pathway breakdowns with the in/out endpoints
    in_out_labels = ["in"] + cluster_names + ["out"]
    for i, (title, cmap) in enumerate(zip(title_list, cmap_list)):
        m = pathways["conn"][i]
        tot = m.sum()
        plot_mat = m / m.sum(1)[:, np.newaxis]
        fig = _heatmatrix(plot_mat, in_out_labels, cmap=cmap, height=600, width=600)
        fig.update_layout(title=f"{title} weights: {tot: .0f}",
                          xaxis_title="post", yaxis_title="pre")
        fig.update_layout(xaxis_tickangle=0)
        fig.update_yaxes(autorange="reversed")
        fig.update_traces(colorbar=dict(tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"]),
                          zmin=0, zmax=1)
        figs.append((fig, f"{dataset}_outrel_conn_matrix_{title}.pdf"))

    for i, (title, cmap) in enumerate(zip(title_list, cmap_list)):
        m = pathways["conn"][i] - np.diag(pathways["diag"][i])
        tot = m.sum()
        plot_mat = m / m.sum(1)[:, np.newaxis]
        fig = _heatmatrix(plot_mat, in_out_labels, cmap=cmap, height=600, width=600)
        fig.update_layout(title=f"{title} weights: {tot: .0f}",
                          xaxis_title="post", yaxis_title="pre")
        fig.update_layout(xaxis_tickangle=0)
        fig.update_yaxes(autorange="reversed")
        fig.update_traces(colorbar=dict(tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"]),
                          zmin=0, zmax=1)
        figs.append((fig, f"{dataset}_outrel_conn_matrix_{title}_wo_selfconn.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_clusters_sankey(
    sankey: dict, colored_main_groups_df: pd.DataFrame,
    colored_seed_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Sankey diagram of inputs → OL internal/output clusters by hit_bin → CB / left OL.

    Consumes the cached output of `pre.get_ol_sankey_connectivity()`. Node
    positions are computed from the diagram label (`OLO{j}.{i}` → x by `i`,
    y by `j`) so the layout adapts to the actual cluster count.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    conn = sankey["conn_diagram"]
    meta_sub = sankey["meta_sub"]
    n_clu = sankey["n_clu"]
    weight_thre = sankey["weight_thre"]

    edges = conn.unstack().reset_index(name="weight").rename(
        columns={"level_0": "post", "level_1": "pre"},
    )
    edges = edges[edges.weight >= weight_thre].reset_index(drop=True)

    # ensure each OLO node has at least one in/out edge so Sankey doesn't drop it
    for i in range(3):
        for j in range(n_clu):
            label = f"OLO{j + 1}.{i + 1}"
            n_in = (edges["post"] == label).sum()
            n_out = (edges["pre"] == label).sum()
            if n_in == n_out == 0:
                continue
            if n_in == 0 and label in conn.columns:
                src = conn[label].idxmax()
                edges = pd.concat([edges, pd.DataFrame(
                    {"post": [label], "pre": [src], "weight": [conn.loc[src, label]]}
                )], ignore_index=True)
            if n_out == 0 and label in conn.index:
                dst = conn.loc[label].idxmax()
                edges = pd.concat([edges, pd.DataFrame(
                    {"post": [dst], "pre": [label], "weight": [conn.loc[label, dst]]}
                )], ignore_index=True)

    diagram_names = list(set(edges["pre"].unique()) | set(edges["post"].unique()))

    # node colors: use main_group for OLI/OLO; seed colors for inputs; region colors for CB / left OL
    diagram_to_mg = dict(meta_sub[["diagram", "main_groups"]].drop_duplicates().values)
    node_color = {}
    for d in diagram_names:
        if d.startswith("OLO"):
            node_color[d] = "#64a0d1"  # OL output blue (overridden below per cluster)
        elif d.startswith("OLI"):
            node_color[d] = colored_main_groups_df.loc[diagram_to_mg.get(d, "OL internal"), "color"]
        elif d in colored_seed_df.index:
            node_color[d] = colored_seed_df.loc[d, "color"]
        elif d in colored_region_df.index:
            node_color[d] = colored_region_df.loc[d, "color"]
        else:
            node_color[d] = "#cccccc"
    # cluster-specific blues for OLO nodes
    cluster_color_map = _cluster_color_df(n_clu)["color"].to_dict()
    for d in diagram_names:
        if d.startswith("OLO"):
            j = int(d[3:].split(".")[0])
            node_color[d] = cluster_color_map[f"oc{j}"]

    # programmatic positions: OL bins 0..3 + merged terminal box at bin 4.
    x_for_bin = {0: 0.05, 1: 0.275, 2: 0.5, 3: 0.725, 4: 0.95}
    seed_y = {
        "L1": 0.95, "L2": 0.83, "L3": 0.71, "R7": 0.59, "R8": 0.47,
        "R7d": 0.35, "R8d": 0.23, "HBeyelet": 0.11,
    }
    region_y = {"CB": 0.5}
    pos_x, pos_y = [], []
    for d in diagram_names:
        if d in seed_y:
            pos_x.append(0.05); pos_y.append(1 - seed_y[d])
        elif d in region_y:
            pos_x.append(0.95); pos_y.append(1 - region_y[d])
        elif d.startswith("OLI."):
            i = int(d.split(".")[1])
            pos_x.append(x_for_bin[i]); pos_y.append(0.2)
        elif d.startswith("OLO"):
            j, i = d[3:].split(".")
            pos_x.append(x_for_bin[int(i)])
            pos_y.append(0.4 + (int(j) - 1) / max(n_clu - 1, 1) * 0.55)
        else:
            pos_x.append(0.5); pos_y.append(0.5)

    node_index = {name: i for i, name in enumerate(diagram_names)}
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=["" for _ in diagram_names],
            color=[node_color[d] for d in diagram_names],
            x=pos_x, y=pos_y,
        ),
        link=dict(
            source=edges["pre"].map(node_index),
            target=edges["post"].map(node_index),
            value=edges["weight"],
            color="rgba(123, 123, 123, 0.4)",
        ),
    ))
    fig.update_layout(width=800, height=450, font=dict(size=18, color="black", family="Arial"))
    _save(fig, save_dir, f"{dataset}_sankey_pathways_w_OLL.pdf")
    return fig


def plot_visual_system_sankey(
    sankey: dict, colored_main_groups_df: pd.DataFrame,
    colored_seed_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    filename: str | None = None, show_labels: bool = True,
):
    """Sankey diagram from visual inputs through OL/VPN and CB/VCBN groups."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    conn = sankey["conn_diagram"]
    node_table = sankey["node_table"].copy().set_index("diagram")
    weight_thre = sankey["weight_thre"]

    raw_edges = conn.unstack().reset_index(name="weight").rename(
        columns={"level_0": "post", "level_1": "pre"},
    )
    raw_edges = raw_edges[raw_edges["weight"] > 0].reset_index(drop=True)

    visible_nodes = set(node_table.index[node_table["plot_node"]])
    display_layer = node_table["display_layer"].to_dict()

    def _remap_target(diagram: str) -> str | None:
        if diagram in visible_nodes:
            return diagram
        if diagram not in node_table.index:
            return None

        row = node_table.loc[diagram]
        category = row["category"]
        candidates = node_table[node_table["plot_node"] & node_table["category"].eq(category)].copy()
        if category in {"VPN", "VCBN"}:
            candidates = candidates[candidates["cluster"].eq(row["cluster"])]
        if candidates.empty:
            return None

        candidates["_layer_dist"] = (
            candidates["display_layer"].astype(float) - float(row["display_layer"])
        ).abs()
        candidates["_neuron_rank"] = -candidates["n_neurons"].astype(float)
        candidates = candidates.sort_values(
            ["_layer_dist", "display_layer", "_neuron_rank"],
            kind="stable",
        )
        return str(candidates.index[0])

    remap = {diagram: _remap_target(diagram) for diagram in node_table.index}
    raw_edges["pre"] = raw_edges["pre"].map(remap)
    raw_edges["post"] = raw_edges["post"].map(remap)
    raw_edges = raw_edges.dropna(subset=["pre", "post"])
    raw_edges = raw_edges[raw_edges["pre"] != raw_edges["post"]]
    raw_edges = (
        raw_edges.groupby(["post", "pre"], as_index=False, sort=False)["weight"].sum()
    )
    raw_edges = raw_edges[
        raw_edges["pre"].map(display_layer) < raw_edges["post"].map(display_layer)
    ].reset_index(drop=True)

    edges = raw_edges[raw_edges["weight"] >= weight_thre].copy().reset_index(drop=True)
    for label in sorted(visible_nodes, key=lambda d: (display_layer[d], d)):
        has_available = (raw_edges["pre"].eq(label) | raw_edges["post"].eq(label)).any()
        if not has_available:
            continue
        if not edges["post"].eq(label).any():
            incoming_i = raw_edges[raw_edges["post"].eq(label)]
            if not incoming_i.empty:
                edges = pd.concat(
                    [edges, incoming_i.nlargest(1, "weight")],
                    ignore_index=True,
                )
        if not edges["pre"].eq(label).any():
            outgoing_i = raw_edges[raw_edges["pre"].eq(label)]
            if not outgoing_i.empty:
                edges = pd.concat(
                    [edges, outgoing_i.nlargest(1, "weight")],
                    ignore_index=True,
                )
    edges = (
        edges.groupby(["post", "pre"], as_index=False, sort=False)["weight"].sum()
    )
    if edges.empty:
        raise ValueError("No Sankey edges remain after plotting filters.")

    nodes = node_table.loc[sorted(visible_nodes)].copy()
    merged_counts = node_table.groupby(node_table.index.map(remap))["n_neurons"].sum()
    merged_counts = merged_counts.drop(index=[None], errors="ignore")
    nodes["n_neurons"] = merged_counts.reindex(nodes.index).fillna(nodes["n_neurons"]).astype(int)
    final_layer_mask = nodes["category"].isin(["left OL", "VNC"])
    nodes.loc[final_layer_mask, "display_layer"] = 6
    seed_order = {"L1": 0, "L2": 1, "L3": 2, "R7": 3, "R8": 4}
    category_order = {
        "visual input": 0, "VIN": 1, "CB": 2, "VPN": 3,
        "VCBN": 4, "left OL": 5, "VNC": 6,
    }

    def _sort_key(row):
        diagram = row.name
        category = row["category"]
        if category == "visual input":
            return (category_order[category], seed_order.get(diagram, 99), diagram)
        cluster = row["cluster"]
        cluster = int(cluster) if pd.notna(cluster) else 0
        return (category_order.get(category, 99), cluster, diagram)

    sort_keys = nodes.apply(_sort_key, axis=1, result_type="expand")
    nodes[["_sort_group", "_sort_cluster", "_sort_name"]] = sort_keys
    nodes = nodes.sort_values(
        ["display_layer", "_sort_group", "_sort_cluster", "_sort_name"],
        kind="stable",
    )
    diagram_names = list(nodes.index)

    def _cluster_palette(prefix: str, n_clu: int, colors: list[str]) -> dict[str, str]:
        if n_clu <= len(colors):
            palette = colors[:n_clu]
        else:
            palette = (colors * ((n_clu // len(colors)) + 1))[:n_clu]
        return {f"{prefix}{i + 1}": color for i, color in enumerate(palette)}

    n_ol_clu = max(int(sankey.get("n_ol_clu", 1)), 1)
    n_cb_clu = max(int(sankey.get("n_cb_clu", 1)), 1)
    ol_colors = _cluster_palette("oc", n_ol_clu, _CLUSTER_BLUES)
    vcbn_colors = _cluster_palette(
        "cc",
        n_cb_clu,
        ["#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a",
         "#fc9272", "#fcbba1", "#fee0d2", "#fff5f0", "#ead7c0"],
    )

    def _node_color(diagram, row):
        category = row["category"]
        if category == "visual input" and diagram in colored_seed_df.index:
            return colored_seed_df.loc[diagram, "color"]
        if category == "VIN":
            return colored_main_groups_df.loc["OL internal", "color"]
        if category == "VPN":
            return ol_colors.get(f"oc{int(row['cluster'])}", "#64a0d1")
        if category == "VCBN":
            return vcbn_colors.get(f"cc{int(row['cluster'])}", "#cb181d")
        if category == "CB":
            return colored_region_df.loc["CB", "color"]
        if category == "left OL":
            return colored_region_df.loc["left OL", "color"]
        if category == "VNC":
            return colored_region_df.loc["VNC", "color"]
        return "#cccccc"

    max_display_layer = max(6, int(nodes["display_layer"].max()))
    x_lookup = {
        layer: 0.06 + float(layer) * 0.82 / max_display_layer
        for layer in sorted(nodes["display_layer"].unique())
    }
    pos_x = [x_lookup[layer] for layer in nodes["display_layer"]]
    pos_y = pd.Series(index=nodes.index, dtype=float)

    max_nodes_per_layer = int(nodes.groupby("display_layer").size().max())
    height = 800
    node_pad = 18
    plot_height = max(1, height - 80)
    pad_norm = node_pad / plot_height

    incoming = edges.groupby("post")["weight"].sum()
    outgoing = edges.groupby("pre")["weight"].sum()
    node_value = pd.Series(0.0, index=nodes.index)
    node_value = node_value.combine(incoming.reindex(nodes.index).fillna(0), max)
    node_value = node_value.combine(outgoing.reindex(nodes.index).fillna(0), max)

    scale_candidates = []
    for _, group in nodes.groupby("display_layer", sort=False):
        total = float(node_value.loc[group.index].sum())
        if total <= 0:
            continue
        available = max(0.01, 1.0 - pad_norm * (len(group) - 1))
        scale_candidates.append(available / total)
    value_to_y = min(scale_candidates) if scale_candidates else 0.0
    node_height = node_value * value_to_y
    vin_overlay_scale = 0.5
    visible_node_height = node_height.copy()
    vin_nodes_all = nodes.index[nodes["category"].eq("VIN")]
    visible_node_height.loc[vin_nodes_all] = (
        node_height.loc[vin_nodes_all] * vin_overlay_scale
    ).clip(lower=0.025)

    def _stack_nodes(index, *, y_min=0.06, y_max=0.94):
        index = list(index)
        if not index:
            return
        if len(index) == 1:
            top = y_min + max(
                0, (y_max - y_min - float(visible_node_height.loc[index[0]])) / 2,
            )
            pos_y.loc[index[0]] = top
            return

        heights = visible_node_height.loc[index].astype(float)
        available = y_max - y_min
        min_gap = pad_norm * 1.05
        extra = available - float(heights.sum()) - min_gap * (len(index) - 1)
        gap = min_gap + max(0, extra) / (len(index) - 1)
        if extra < 0:
            gap = max(0, (available - float(heights.sum())) / (len(index) - 1))

        y = y_min
        for diagram in index:
            pos_y.loc[diagram] = y
            y += float(visible_node_height.loc[diagram]) + gap

    for _, group in nodes.groupby("display_layer", sort=False):
        if group.index.isin(["left OL", "CB.6", "VNC"]).any():
            final_order = [d for d in ["left OL", "CB.6", "VNC"] if d in group.index]
            _stack_nodes(final_order, y_min=0.12, y_max=0.82)
            leftovers = [d for d in group.index if d not in final_order]
            _stack_nodes(leftovers, y_min=0.84, y_max=0.94)
            continue

        y_min = 0.34 if "VIN.1" in group.index else 0.02
        _stack_nodes(group.index, y_min=y_min, y_max=0.98)

    pos_y = pos_y.fillna(0.5).clip(lower=0.02, upper=0.98)
    node_colors = [_node_color(d, nodes.loc[d]) for d in diagram_names]
    node_width = 0.012
    node_geom = pd.DataFrame({
        "x0": [x_lookup[nodes.loc[d, "display_layer"]] for d in diagram_names],
        "y0": [float(pos_y.loc[d]) for d in diagram_names],
        "height": [float(visible_node_height.loc[d]) for d in diagram_names],
        "color": node_colors,
    }, index=diagram_names)
    node_geom["x1"] = node_geom["x0"] + node_width
    node_geom["y1"] = node_geom["y0"] + node_geom["height"]

    edge_draw = edges.copy()
    edge_draw["source_width"] = edge_draw["weight"] * value_to_y
    edge_draw["target_width"] = edge_draw["weight"] * value_to_y
    edge_draw.loc[edge_draw["pre"].map(nodes["category"]).eq("VIN"), "source_width"] *= vin_overlay_scale
    edge_draw.loc[edge_draw["post"].map(nodes["category"]).eq("VIN"), "target_width"] *= vin_overlay_scale

    edge_draw["_pre_y"] = edge_draw["pre"].map(node_geom["y0"])
    edge_draw["_post_y"] = edge_draw["post"].map(node_geom["y0"])

    source_slots = {}
    for node, group_i in edge_draw.sort_values(["_post_y", "post"]).groupby("pre", sort=False):
        total_width = float(group_i["source_width"].sum())
        y = float(node_geom.loc[node, "y0"]) + max(0, float(node_geom.loc[node, "height"]) - total_width) / 2
        for idx, edge_i in group_i.iterrows():
            width_i = float(edge_i["source_width"])
            source_slots[idx] = (y, y + width_i)
            y += width_i

    target_slots = {}
    for node, group_i in edge_draw.sort_values(["_pre_y", "pre"]).groupby("post", sort=False):
        total_width = float(group_i["target_width"].sum())
        y = float(node_geom.loc[node, "y0"]) + max(0, float(node_geom.loc[node, "height"]) - total_width) / 2
        for idx, edge_i in group_i.iterrows():
            width_i = float(edge_i["target_width"])
            target_slots[idx] = (y, y + width_i)
            y += width_i

    fig = go.Figure()
    for idx, edge_i in edge_draw.iterrows():
        sx = float(node_geom.loc[edge_i["pre"], "x1"])
        tx = float(node_geom.loc[edge_i["post"], "x0"])
        sy0, sy1 = source_slots[idx]
        ty0, ty1 = target_slots[idx]
        curve = max(0.02, abs(tx - sx) * 0.45)
        path = (
            f"M {sx},{sy0} "
            f"C {sx + curve},{sy0} {tx - curve},{ty0} {tx},{ty0} "
            f"L {tx},{ty1} "
            f"C {tx - curve},{ty1} {sx + curve},{sy1} {sx},{sy1} Z"
        )
        fig.add_shape(
            type="path", path=path,
            xref="x", yref="y",
            fillcolor="rgba(123, 123, 123, 0.35)",
            line=dict(width=0, color="rgba(123, 123, 123, 0)"),
            layer="below",
        )

    for diagram in diagram_names:
        geom = node_geom.loc[diagram]
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=float(geom["x0"]), x1=float(geom["x1"]),
            y0=float(geom["y0"]), y1=float(geom["y1"]),
            fillcolor=str(geom["color"]),
            line=dict(color="black", width=0.5),
            layer="above",
        )
        fig.add_trace(go.Scatter(
            x=[(float(geom["x0"]) + float(geom["x1"])) / 2],
            y=[(float(geom["y0"]) + float(geom["y1"])) / 2],
            mode="markers",
            marker=dict(size=1, opacity=0),
            hovertemplate=(
                f"{diagram}<br>{int(nodes.loc[diagram, 'n_neurons'])} neurons"
                "<extra></extra>"
            ),
            showlegend=False,
        ))
        if show_labels:
            fig.add_annotation(
                xref="x", yref="y",
                x=float(geom["x1"]) + 0.004,
                y=(float(geom["y0"]) + float(geom["y1"])) / 2,
                text=diagram,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=12, color="black", family="Arial"),
            )

    fig.update_layout(
        width=1000, height=height,
        font=dict(size=12, color="black", family="Arial"),
        margin=dict(l=40, r=100, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(range=[0, 1], visible=False, fixedrange=True)
    fig.update_yaxes(range=[1, 0], visible=False, fixedrange=True)
    out_name = filename or f"{dataset}_visual_system_sankey.pdf"
    _save(fig, save_dir, out_name)
    return fig


def _save_graph_html(g, layout, vertex_colors, vertex_size, out_path: Path) -> None:
    """Save an igraph graph as an interactive Plotly HTML with node-name hover."""
    if not _WRITE_HTML:
        return

    coords = layout.coords
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    edge_x, edge_y = [], []
    for src, tgt in g.get_edgelist():
        edge_x += [xs[src], xs[tgt], None]
        edge_y += [ys[src], ys[tgt], None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="lightgray", width=0.8),
        hoverinfo="none",
    )
    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(
            color=vertex_colors,
            size=vertex_size,
            line=dict(color="black", width=0.5),
        ),
        text=g.vs["name"],
        hoverinfo="text",
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=10, r=10, t=10, b=10),
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


# %%
def plot_clusters_graph(
    type_part_df: pd.DataFrame, ol_meta: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    colored_seed_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    edge_color: str = "lightgray",
):
    """igraph network of cell-type / cluster participation (visual input + OL
    internal). Edges are undirected gray lines; `colored_sign_df` is accepted
    for API compatibility but no longer used for edge colouring."""
    import igraph as ig
    import matplotlib.pyplot as plt
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    save_dir.mkdir(parents=True, exist_ok=True)

    cluster_names_c = list(type_part_df.columns[type_part_df.columns.str.contains(r"^c\d+$")].values)
    cluster_names_no_c = [s[1:] for s in cluster_names_c]
    n_clu = len(cluster_names_c)

    base = type_part_df.merge(
        ol_meta[["instance", "main_groups", "sign"]].drop_duplicates(),
        on="instance", how="left",
    )
    base = base[base.main_groups.isin(["visual input", "OL internal"])].reset_index(drop=True)
    long = base.melt(
        id_vars=["instance", "main_groups", "sign"],
        value_vars=cluster_names_c, var_name="post", value_name="weight",
    ).rename(columns={"instance": "pre"})
    long = long[long.weight >= 1 / n_clu].copy()
    long["pre"] = [s.split("_")[0] for s in long["pre"]]
    long["post"] = [s[1:] for s in long["post"]]

    nodes = list(set(long["pre"]).union(long["post"]))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_to_idx[s], node_to_idx[t]) for s, t in zip(long["pre"], long["post"])]

    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    g.add_edges(edges)
    g.vs["name"] = nodes
    g.es["weight"] = long["weight"].tolist()

    cluster_color_map = _cluster_color_df(n_clu)["color"].to_dict()
    cluster_to_color = {f"{i + 1}": cluster_color_map[f"oc{i + 1}"] for i in range(n_clu)}

    pre_color = (
        long.assign(color=long["main_groups"].map(colored_main_groups_df["color"]))
            .set_index("pre")["color"].to_dict()
    )
    pre_color.update(colored_seed_df["color"].to_dict())
    pre_color.update(cluster_to_color)
    vertex_colors = [pre_color.get(v["name"], "gray") for v in g.vs]

    highlight = set(cluster_names_no_c) | {s for s in _PART_TYPES}
    vertex_label = [s if s in highlight else "" for s in g.vs["name"]]
    vertex_frame_color = ["black" if v["name"] in highlight else "white" for v in g.vs]
    vertex_size = [25 if v["name"][0].isdigit() else 15 for v in g.vs]

    layout = g.layout("kk")
    fig, ax = plt.subplots(figsize=(8, 8))
    ig.plot(
        g, target=ax, layout=layout,
        vertex_label=vertex_label, vertex_label_size=14,
        vertex_color=vertex_colors, vertex_frame_color=vertex_frame_color,
        vertex_frame_width=1, vertex_size=vertex_size,
        edge_color=edge_color, edge_arrow_size=0.0,
        edge_width=g.es["weight"],
    )
    ax.axis("off")
    out = save_dir / f"{dataset}_graph_pathways_w_OLL_v1.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _save_graph_html(
        g, layout, vertex_colors, vertex_size,
        HTML_FIG_DIR / out.with_suffix(".html").name,
    )
    return out


# %% [markdown]
# === 4. LR clustering ===


# %%
def plot_lr_clustering_sweep(
    sweep_df: pd.DataFrame, n0: int, *,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Six line/scatter PDFs for the L/R clustering threshold sweep.

    `n0` is the reference cluster count (from the chosen FRAC in section 3);
    used to draw dashed reference lines on each panel.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "clustering"
    idx0 = int(np.where(sweep_df["n_clu_R"].values == n0)[0][0])
    figs = []

    # panel 1: n_R vs frac
    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["frac"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.update_layout(xaxis_title="No. R clusters", yaxis_title="Clustering threshold")
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0, 2])
    fig.update_yaxes(type="log", tickvals=[0.05, 0.1, 0.5, 1], range=[-1.7, 0.1])
    figs.append((fig, f"{dataset}_no_thre.pdf"))

    # panel 2: diag vs n_R
    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["diag"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.update_layout(
        xaxis_title="no. clusters",
        yaxis_title="frac. diagonal cluster-cluster connectivity",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(tickvals=[0, 0.5, 1], range=[0, 1])
    figs.append((fig, f"{dataset}_clustering_threshold_diag.pdf"))

    # panel 3: n_L vs n_R
    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["n_clu_L"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.add_trace(go.Scatter(x=[1, 300], y=[1, 300], mode="lines",
                             line=dict(color="gray", dash="dash"), showlegend=False))
    fig.update_layout(
        xaxis_title="no. R clusters", yaxis_title="no. L clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    figs.append((fig, f"{dataset}_LR_no_clusters.pdf"))

    # panel 4: ARI_LR vs n_R
    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["ari_LR"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.update_layout(
        xaxis_title="no. R clusters", yaxis_title="ARI between L and R clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_LR_ARI.pdf"))

    # panel 5: ARI_fromR vs n_R
    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["ari_fromR"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0])
    fig.update_layout(
        xaxis_title="no. R clusters", yaxis_title="ARI between L and L from R clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_LR_fromR_ARI.pdf"))

    # panel 6: ARI comparison (ari_fromR x, ari_LR y)
    df = pd.DataFrame({"x": sweep_df["ari_fromR"].values, "y": sweep_df["ari_LR"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=500)
    fig.add_trace(go.Scatter(x=[0.01, 1], y=[0.01, 1], mode="lines",
                             line=dict(color="gray", dash="dash"), showlegend=False))
    fig.update_layout(
        yaxis_title="ARI between L and R clusters",
        xaxis_title="ARI between L and L from R clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_LR_ARI_comparison.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_lr_cluster_confusion(
    match: dict, *,
    dataset: str = DATASET, save_dir: Path | None = None, anno_min: int = 10,
    filename_stem: str = "left_to_right_confusion",
    xaxis_title: str = "right cluster", yaxis_title: str = "left cluster",
):
    """Heatmap of the L↔R cluster confusion matrix (anno only for cells ≥ anno_min)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "clustering"
    conf = match["confusion"]
    tick_labels = match["tick_labels"]
    anno = np.where(conf >= anno_min, conf.astype(object), "")
    fig = _heatmatrix(conf, tick_labels, anno=anno)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    _save(fig, save_dir, f"{dataset}_{filename_stem}.pdf")
    return fig


# %% [markdown]
# === 5. Polarity experimental comparisons ===


# %%
_POLARITY_DISPLAY_RENAMES = {"5thsLNv_LNd6": "5thsLNv"}


# %%
def plot_polarity_stacked_bars(
    comparison: pd.DataFrame, colored_seed_df: pd.DataFrame,
    *, dataset: str = DATASET, filename_stem: str = "polarity_predictions",
    save_dir: Path | None = None, width: int | None = None, height: int = 350,
    font_size: int = 18,
):
    """Stacked bars of normalised visual-input effective weight per instance,
    colored by visual-input seed. Instance labels on the x-axis are bolded
    when the ground-truth `polarity` column is `'OFF'` (legacy convention).
    Display names are shortened via `_POLARITY_DISPLAY_RENAMES` for ticktext
    only — the underlying `out type` category keeps the full name so the bar
    ordering is unambiguous. Width defaults to ~35 px per instance + padding."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "exp_comparison"
    if width is None:
        width = int(max(400, 35 * len(comparison) + 150) * 0.8)

    # `colored_seed_df` index is bare seed names (no suffix); match columns in the comparison
    suffix = "_R" if any(c.endswith("_R") for c in comparison.columns) else "_L"
    seed_cols = [s + suffix for s in colored_seed_df.index if s + suffix in comparison.columns]

    long = comparison[["instance"] + seed_cols].copy()
    long["out type"] = long["instance"].str[:-2]
    melted = long.melt(
        id_vars=["out type"], value_vars=seed_cols,
        var_name="in type", value_name="effective weight",
    )
    melted["in type"] = melted["in type"].str[:-2]

    cmap = colored_seed_df["color"].to_dict()
    fig = _stacked_bars(
        melted, "out type", "effective weight", "in type",
        cmap, xrot=-45, width=width, height=height, font_size=font_size,
    )

    order = comparison["instance"].str[:-2].tolist()
    labels = [
        f"<b>{_POLARITY_DISPLAY_RENAMES.get(t, t)}</b>" if p == "OFF"
        else _POLARITY_DISPLAY_RENAMES.get(t, t)
        for t, p in zip(order, comparison["polarity"].values)
    ]
    fig.update_xaxes(
        categoryorder="array", categoryarray=order,
        tickvals=order, ticktext=labels,
    )
    _save(fig, save_dir, f"{dataset}_{filename_stem}.pdf", html=True)
    return fig


# %% [markdown]
# === 6. Propagated RFs ===


# %%
_RF_EXAMPLE_BID = 30134
_RF_STAR_EXAMPLES = [
    "LPT111", "VS", "aMe12",
    "LC4", "LC6", "LPLC2", "T4a", "Tm5a",
]
_RF_STAR_OUTPUT = _RF_STAR_EXAMPLES
_RF_STAR_INTERNAL = _RF_STAR_EXAMPLES


# %%
def plot_rf_example(
    ol_meta: pd.DataFrame, stepsn, params_all_df: pd.DataFrame,
    example_bid: int, in_instances: list[str],
    *, dataset: str = DATASET, save_dir: Path | None = None,
    per_input_threshold: float = 0.2,
):
    """Three figures for one example OL output neuron:
    (1) original effective-weight hex RF,
    (2) fitted 2D-Gaussian RF + overlay,
    (3) per-visual-input hex RFs for inputs above `per_input_threshold * tot_max`.
    """
    from connectome_interpreter.external_map import hex_heatmap
    from utils.external_rf import twoD_Gaussian
    from utils.ol_rf import compute_rf

    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    figs = []
    inidx = ol_meta[ol_meta.instance.isin(in_instances)].idx.values
    outidx = ol_meta[ol_meta.bodyId == example_bid].idx.values

    df = compute_rf(ol_meta, stepsn, inidx, outidx)
    if df.empty:
        return figs
    tot_max = float(np.abs(df["effective weight"]).max())
    orig = df.set_index("coords")[["effective weight"]]
    fig = hex_heatmap(
        orig, custom_colorscale="RdBu_r",
        global_min=-tot_max, global_max=tot_max, dataset="mcns_right",
    )
    figs.append((fig, f"{dataset}_example_{example_bid}_rf_orig.pdf"))

    row = params_all_df.loc[params_all_df["bodyId"] == example_bid]
    if not row.empty:
        row = row.iloc[0]
        fit = twoD_Gaussian(df["x"].values, df["y"].values,
                            row["x0"], row["y0"], row["a"], row["b"], row["phi"])
        fit_df = pd.DataFrame(fit * tot_max, index=df["coords"], columns=["effective weight"])
        fit_df = fit_df[~fit_df.index.duplicated()]
        fig = hex_heatmap(
            fit_df, custom_colorscale="RdBu_r",
            global_min=-tot_max, global_max=tot_max, dataset="mcns_right",
        )
        _add_gaussian_ellipses(
            fig, pd.DataFrame([row]), example_bid=example_bid, y_scale=np.sqrt(3),
        )
        figs.append((fig, f"{dataset}_example_{example_bid}_rf_fit.pdf"))

    for inst in in_instances:
        ii = ol_meta[ol_meta.instance == inst].idx.values
        d_i = compute_rf(ol_meta, stepsn, ii, outidx)
        if d_i.empty or np.abs(d_i["effective weight"]).max() < per_input_threshold * tot_max:
            continue
        plot_df = d_i.set_index("coords")[["effective weight"]]
        fig = hex_heatmap(
            plot_df, custom_colorscale="RdBu_r",
            global_min=-tot_max, global_max=tot_max, dataset="mcns_right",
        )
        figs.append((fig, f"{dataset}_example_{example_bid}_rf_{inst[:-2]}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_rf_positions_across_neurons(
    params_all_df: pd.DataFrame, instance: str,
    *, dataset: str = DATASET, example_bid: int | None = None,
    save_dir: Path | None = None,
):
    """Gaussian-ellipse field for every fit in one instance, overlaid on a blank
    hex grid. Also draws an R^2 hex heatmap for the same instance.
    """
    from connectome_interpreter.external_map import hex_heatmap
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    figs = []
    sub = params_all_df[params_all_df["instance"] == instance].copy()
    if sub.empty:
        return figs

    blank = pd.DataFrame([0], index=["0,22"], columns=["value"])
    fig = hex_heatmap(
        blank, custom_colorscale="Picnic",
        global_min=-1, global_max=1, dataset="mcns_right",
    )
    _add_gaussian_ellipses(fig, sub, example_bid=example_bid, y_scale=np.sqrt(3))
    fig.update_layout(font=dict(family="arial", size=22))
    figs.append((fig, f"{dataset}_{instance}_rf_pos_par.pdf"))

    r2 = pd.DataFrame({
        "x": sub["x0"].values, "y": sub["y0"].values * np.sqrt(3), "value": sub["r2"].values,
    })
    r2["coords"] = r2.apply(lambda r: f"{r['x']:.0f},{r['y']:.0f}", axis=1)
    cmax = float(min(np.quantile(np.abs(r2["value"]), 0.95), 1))
    fig = hex_heatmap(
        r2.set_index("coords")[["value"]],
        custom_colorscale="RdBu", global_min=-cmax, global_max=cmax,
        dataset="mcns_right",
    )
    fig.update_layout(title="R2", title_x=0.1, title_y=0.95)
    figs.append((fig, f"{dataset}_{instance}_rf_r2.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_input_rf_example(
    roi_counts: pd.DataFrame, input_raw: pd.DataFrame,
    example_bid: int, target_instance: str,
    *, roi: str = "LO(R)", dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """Three direct-input RF figures for one example body (legacy
    `find_direct_rfs.ipynb` cells 15/18/22):
    (1) per-column post-synapse hex RF, max-normalised,
    (2) fitted 2D-Gaussian hex RF + ellipse overlay,
    (3) ellipse field over all `target_instance` fits in `roi`.

    `roi_counts` must have ROIs of the form `<ROI>_col_<h1>_<h2>` (e.g.
    `LO_R_col_18_19`) providing per-column post counts. `input_raw` is per-body
    direct-input fits from `get_input_rf_raw_ol`.
    """
    from connectome_interpreter.external_map import hex_heatmap, load_dataset
    from utils.external_rf import twoD_Gaussian

    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    figs = []

    roi_prefix = roi.replace("(", "_").replace(")", "")
    col_re = re.compile(rf"^{re.escape(roi_prefix)}_col_(-?\d+)_(-?\d+)$")

    hex_df = load_dataset("Nern2024")[["x", "y"]].drop_duplicates().reset_index(drop=True)

    body_rows = roi_counts[roi_counts["bodyId"] == example_bid]
    parsed = [
        (int(m.group(1)), int(m.group(2)), p)
        for r, p in zip(body_rows["roi"], body_rows["post"])
        if (m := col_re.match(str(r)))
    ]
    col_df = pd.DataFrame(parsed, columns=["hex1_id", "hex2_id", "post"])
    col_df["x"] = col_df["hex2_id"] - col_df["hex1_id"]
    col_df["y"] = col_df["hex2_id"] + col_df["hex1_id"]
    df = col_df.merge(hex_df, how="right", on=["x", "y"]).fillna({"post": 0})
    df["coords"] = df.apply(lambda r: f"{int(r['x'])},{int(r['y'])}", axis=1)
    amax = df["post"].max()
    df["effective weight"] = df["post"] / amax if amax > 0 else 0.0

    orig = df.set_index("coords")[["effective weight"]]
    orig = orig[~orig.index.duplicated()]
    fig = hex_heatmap(
        orig, custom_colorscale="RdBu_r",
        global_min=-1, global_max=1, dataset="mcns_right",
    )
    figs.append((fig, f"{dataset}_{target_instance}_input_rf_orig.pdf"))

    row = input_raw[(input_raw["bodyId"] == example_bid) & (input_raw["roi"] == roi)]
    if not row.empty:
        row = row.iloc[0]
        fit = twoD_Gaussian(
            df["x"].values.astype(float), (df["y"].values / np.sqrt(3)).astype(float),
            row["x0"], row["y0"], row["a"], row["b"], row["phi"],
        )
        fit_df = pd.DataFrame({"effective weight": fit}, index=df["coords"])
        fit_df = fit_df[~fit_df.index.duplicated()]
        fig = hex_heatmap(
            fit_df, custom_colorscale="RdBu_r",
            global_min=-1, global_max=1, dataset="mcns_right",
        )
        _add_gaussian_ellipses(
            fig, pd.DataFrame([row]), example_bid=example_bid, y_scale=np.sqrt(3),
        )
        figs.append((fig, f"{dataset}_{target_instance}_input_rf_fit.pdf"))

    sub = input_raw[(input_raw["instance"] == target_instance) & (input_raw["roi"] == roi)]
    if not sub.empty:
        blank = pd.DataFrame([0], index=["-16,38"], columns=["value"])
        fig = hex_heatmap(
            blank, custom_colorscale="Picnic",
            global_min=-1, global_max=1, dataset="mcns_right",
        )
        _add_gaussian_ellipses(fig, sub, example_bid=example_bid, y_scale=np.sqrt(3))
        figs.append((fig, f"{dataset}_{target_instance}_input_rf_pos_par.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def _rf_axis_update(fig, col, *, axis="x"):
    """Apply the RF-param-specific axis formatting the legacy used across every
    scatter/box/histogram: log scales for amp/size/n_col, clipped ranges for
    r2/ecc, degree ticks for phi."""
    upd = fig.update_xaxes if axis == "x" else fig.update_yaxes
    if col == "amp":
        upd(title=f"ARF {col}", type="log", range=[np.log10(1e-5), np.log10(1)], tickangle=0)
    elif col == "size":
        upd(title=f"ARF {col}", type="log", tickformat="d", dtick=1,
            range=[np.log10(0.6), np.log10(350)])
    elif col == "n_col":
        upd(title=f"ARF {col}", type="log", tickformat="d", dtick=1,
            range=[np.log10(2), np.log10(1200)])
    elif col == "r2":
        upd(title=f"ARF {col}", range=[-1.1, 1.1], tickangle=0)
    elif col == "ecc":
        upd(title=f"ARF {col}", range=[-0.1, 1.1])
    elif col == "phi":
        upd(title="ARF ori (deg)",
            tickvals=[-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2],
            ticktext=["-90", "-45", "0", "45", "90"],
            range=[-np.pi / 2 * 1.1, np.pi / 2 * 1.1])


# %%
def plot_rf_boxes_by_celltype(
    params_all_df: pd.DataFrame, plot_cell_types: list[str],
    colored_main_groups_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Per-cell-type box plots of r2/amp/size/ecc/n_col/phi, sorted by median
    hitting_time, colored by main_groups."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    sub = params_all_df[params_all_df["instance"].isin(plot_cell_types)].copy()
    if sub.empty:
        return []
    color_map = dict(zip(
        sub["cell_type"],
        colored_main_groups_df.loc[sub["main_groups"], "color"].values,
    ))
    grouped = sub.groupby(["instance", "cell_type"])["hitting_time"].median().reset_index()
    sorted_types = grouped.loc[grouped.sort_values(["hitting_time", "instance"]).index, "cell_type"].values
    sub["cell_type"] = pd.Categorical(sub["cell_type"], categories=sorted_types, ordered=True)
    sub = sub.sort_values("cell_type")

    figs = []
    for val in ["r2", "amp", "size", "ecc", "n_col", "phi"]:
        fig = _box(sub, val, "cell_type", "cell_type", color_map, height=650)
        _rf_axis_update(fig, val, axis="x")
        fig.update_traces(jitter=0.4, marker=dict(size=3, opacity=0.4),
                          selector=dict(type="box"))
        figs.append((fig, f"{dataset}_examples_rf_{val}.pdf"))
    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_rf_scatters_by_main_groups(
    params_type_df: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    *,
    star_types: list[str] | None = None,
    star_types_output: list[str] | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Per-instance scatters colored by main_groups (and by sign): n_neurons vs
    each RF param, plus eight X-Y param pairs (amp/n_col/hitting_time/frac_ff
    vs size/r2/phi/amp). Returns list[(fig, name)]."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    stars_mg = list(star_types) if star_types is not None else [s + "_R" for s in _RF_STAR_INTERNAL]
    stars_sign = list(star_types_output) if star_types_output is not None else [s + "_R" for s in _RF_STAR_OUTPUT]

    sub = params_type_df[
        params_type_df["main_groups"].isin(["OL internal", "OL output", "CB input"])
    ].copy()
    figs = []

    for val in ["r2", "amp", "size", "ecc", "n_col", "phi"]:
        fig = _scatter(
            sub, "n_neurons", val, "main_groups", "instance",
            colored_main_groups_df, stars_mg,
        )
        fig.update_layout(yaxis_title=f"ARF {val}", xaxis_title="no. neurons")
        fig.update_xaxes(type="log", tickformat="d", dtick=1,
                         range=[np.log10(0.6), np.log10(1500)])
        _rf_axis_update(fig, val, axis="y")
        figs.append((fig, f"{dataset}_examples_no_vs_{val}.pdf"))

    pairs = [("amp", "size"), ("n_col", "size"), ("amp", "r2"), ("ecc", "phi"),
             ("hitting_time", "size"), ("frac_ff", "size"),
             ("hitting_time", "amp"), ("frac_ff", "amp"),
             ("r2", "size"), ("phi", "size"), ("VIC", "size")]
    for xv, yv in pairs:
        fig = _scatter(
            sub, xv, yv, "main_groups", "instance",
            colored_main_groups_df, stars_mg,
        )
        fig.update_layout(yaxis_title=f"ARF {yv}", xaxis_title=f"ARF {xv}")
        if xv == "hitting_time":
            fig.update_xaxes(title="layer", range=[0.8, 4.2])
        elif xv == "frac_ff":
            fig.update_xaxes(title="ff fraction", range=[-0.1, 1.1])
        elif xv == "VIC":
            fig.update_xaxes(
                title="VIC", type="log", tickvals=[0.01, 0.1],
                range=[np.log10(0.002), np.log10(0.6)],
            )
        else:
            _rf_axis_update(fig, xv, axis="x")
        _rf_axis_update(fig, yv, axis="y")
        fig.update_xaxes(tickangle=0)
        figs.append((fig, f"{dataset}_examples_{xv}_vs_{yv}.pdf"))

    sign_sub = sub[sub.sign != 0].copy()
    sign_sub["sign"] = sign_sub["sign"].astype(int)
    cmap_sign = colored_sign_df.copy()
    cmap_sign.index = cmap_sign.index.astype(int)
    for xv, yv in [("amp", "size"), ("n_col", "size"), ("amp", "r2"), ("ecc", "phi"),
                   ("hitting_time", "size"), ("frac_ff", "size"),
                   ("hitting_time", "amp"), ("frac_ff", "amp")]:
        fig = _scatter(
            sign_sub, xv, yv, "sign", "instance", cmap_sign, stars_sign,
        )
        fig.update_layout(yaxis_title=f"ARF {yv}", xaxis_title=f"ARF {xv}")
        if xv == "hitting_time":
            fig.update_xaxes(title="layer", range=[0.8, 4.2])
        elif xv == "frac_ff":
            fig.update_xaxes(title="ff fraction", range=[-0.1, 1.1])
        else:
            _rf_axis_update(fig, xv, axis="x")
        _rf_axis_update(fig, yv, axis="y")
        fig.update_xaxes(tickangle=0)
        figs.append((fig, f"{dataset}_examples_{xv}_vs_{yv}_sign.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_rf_scatters_by_cluster(
    params_type_df: pd.DataFrame, in_out_instance_df: pd.DataFrame,
    *, cluster_names: list[str],
    colored_main_groups_df: pd.DataFrame,
    colored_cluster_df: pd.DataFrame | None = None,
    star_types_output: list[str] | None = None,
    star_types_internal: list[str] | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
    jitter: float = 0.2,
):
    """Per-instance RF params jittered against cluster index. Two panels:
    OL-output types (by `idxmax` cluster) and OL-internal types (filtered to
    max-participation > 0.4)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    n_clu = len(cluster_names)
    # Display names add "o" prefix (e.g. "c1" -> "oc1") for this panel only
    display_names = ["o" + n for n in cluster_names]
    if colored_cluster_df is None:
        blues = ["#041f4a", "#08306b", "#08519c", "#2171b5", "#4292c6",
                 "#6baed6", "#9ecae1", "#c6dbef", "#deebf7"]
        colored_cluster_df = pd.DataFrame(blues[:n_clu], index=display_names, columns=["color"])
    else:
        colored_cluster_df = colored_cluster_df.copy()
        colored_cluster_df.index = display_names
    stars_out = list(star_types_output) if star_types_output is not None else list(
        set([s + "_R" for s in _OUT_TYPES]) | set([s + "_R" for s in _CLUSTER_TYPES])
    )
    stars_int = list(star_types_internal) if star_types_internal is not None else [s + "_R" for s in _RF_STAR_INTERNAL]

    figs = []
    rng = np.random.default_rng(0)

    # OL output: y = cluster
    out = (
        params_type_df[params_type_df["main_groups"] == "OL output"]
        .merge(in_out_instance_df[["instance", "idxmax"]], how="inner", on="instance")
    )
    out["cluster_name"] = "o" + out["idxmax"]
    out["cluster_idx"] = out["idxmax"].str.slice(1).astype(int)
    out = out.dropna(subset=["size"]).reset_index(drop=True)
    out["y"] = out["cluster_idx"].astype(float) + rng.uniform(-jitter, jitter, len(out))

    for val in ["r2", "size", "amp", "ecc", "phi"]:
        fig = _scatter(out, val, "y", "cluster_name", "instance",
                       colored_cluster_df, stars_out, height=400, width=550)
        fig.update_layout(yaxis_title="cluster", xaxis_title=f"ARF {val}")
        fig.update_yaxes(autorange="reversed",
                         tickvals=list(range(1, n_clu + 1)),
                         ticktext=display_names, tickangle=0)
        if val == "size":
            fig.update_layout(font=dict(family="arial", size=22), yaxis_title="class")
            fig.update_xaxes(type="log", tickformat="d",
                             tickvals=[5, 10, 50, 100], range=[np.log10(3), np.log10(250)])
        elif val == "amp":
            fig.update_xaxes(type="log", range=[np.log10(1e-4), np.log10(0.05)])
        elif val == "phi":
            fig.update_xaxes(title="ARF ori (deg)",
                             tickvals=[-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2],
                             ticktext=["-90", "-45", "0", "45", "90"])
        elif val == "r2":
            fig.update_xaxes(range=[-1.1, 1.1])
        else:
            fig.update_xaxes(range=[-0.1, 1.1])
        figs.append((fig, f"{dataset}_examples_pathway_vs_{val}_output.pdf"))

    # OL internal: x = cluster
    inside = (
        params_type_df[params_type_df["main_groups"] == "OL internal"]
        .merge(in_out_instance_df, how="left", on="instance")
    )
    inside = inside[inside["max"] > 0.4].dropna(subset=["size"]).reset_index(drop=True)
    inside["cluster_idx"] = inside["idxmax"].str.slice(1).astype(int)
    inside["x"] = inside["cluster_idx"].astype(float) + rng.uniform(-jitter, jitter, len(inside))

    for val in ["r2", "size", "amp", "ecc", "phi"]:
        fig = _scatter(inside, "x", val, "main_groups", "instance",
                       colored_main_groups_df, stars_int, height=400, width=550)
        fig.update_layout(xaxis_title="cluster", yaxis_title=f"ARF {val}")
        fig.update_xaxes(tickvals=list(range(1, n_clu + 1)),
                         ticktext=cluster_names, tickangle=0)
        if val == "size":
            fig.update_yaxes(type="log", tickformat="d", dtick=1,
                             range=[np.log10(0.6), np.log10(350)])
        elif val == "amp":
            fig.update_yaxes(type="log", range=[np.log10(1e-4), np.log10(0.8)])
        elif val == "phi":
            fig.update_yaxes(title="ARF ori (deg)",
                             tickvals=[-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2],
                             ticktext=["-90", "-45", "0", "45", "90"])
        elif val == "r2":
            fig.update_yaxes(range=[-1.1, 1.1])
        else:
            fig.update_yaxes(range=[0, 1])
        figs.append((fig, f"{dataset}_examples_pathway_vs_{val}_internal.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_rf_histograms(
    params_type_df: pd.DataFrame, in_out_instance_df: pd.DataFrame,
    *, cluster_names: list[str],
    colored_main_groups_df: pd.DataFrame,
    colored_sign_df: pd.DataFrame,
    colored_cluster_df: pd.DataFrame | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Size-histogram (by main_groups / sign) + size cumsum by cluster +
    amp/ecc/phi/r2 histograms (by main_groups / sign)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    n_clu = len(cluster_names)
    if colored_cluster_df is None:
        blues = ["#041f4a", "#08306b", "#08519c", "#2171b5", "#4292c6",
                 "#6baed6", "#9ecae1", "#c6dbef", "#deebf7"]
        colored_cluster_df = pd.DataFrame(blues[:n_clu], index=cluster_names, columns=["color"])
    figs = []

    size_bins = 2 ** np.linspace(np.log2(0.6), np.log2(350), 12)
    base = params_type_df[
        params_type_df["main_groups"].isin(["OL internal", "OL output", "CB input"])
    ].copy()

    cmaps = {"main_groups": colored_main_groups_df, "sign": colored_sign_df}
    for grp, cmap in cmaps.items():
        df = base.copy()
        if grp == "sign":
            df = df[df.sign != 0]
            df["sign"] = df["sign"].astype(int)
            cmap = cmap.copy()
            cmap.index = cmap.index.astype(int)
        fig = _hist_lines_v(df, "size", grp, size_bins, cmap, height=400,
                            width=405 if grp == "sign" else 450)
        fig.update_layout(yaxis_title="RF size", xaxis_title="frac. cell types",
                          xaxis_range=[-.01, 0.45])
        fig.update_yaxes(type="log", tickformat="d", dtick=1,
                         range=[np.log10(0.6), np.log10(350)])
        figs.append((fig, f"{dataset}_hist_rf_size_{grp}_types.pdf"))

    # cumsum by cluster (OL output)
    out = (
        params_type_df[params_type_df["main_groups"] == "OL output"]
        .merge(in_out_instance_df[["instance", "idxmax"]], how="inner", on="instance")
    )
    out["cluster_name"] = out["idxmax"]
    cs_bins = 2 ** np.linspace(np.log2(2.5), np.log2(280), 12)
    fig = _hist_cumsum_lines(out, "size", "cluster_name", cs_bins,
                             colored_cluster_df, height=400, width=450)
    fig.update_layout(xaxis_title="ARF size", yaxis_title="cum. frac. cell types",
                      yaxis_range=[-.05, 1.05], yaxis_tickvals=[0, 0.5, 1],
                      xaxis_tickvals=[5, 10, 50, 100],
                      font=dict(family="arial", size=20))
    fig.update_xaxes(type="log", tickformat="d", dtick=1,
                     range=[np.log10(3), np.log10(250)])
    figs.append((fig, f"{dataset}_rf_size_cumsum_types.pdf"))

    # other histograms
    hist_bins = {
        "amp": 10 ** np.linspace(-5, 0, 12),
        "ecc": np.linspace(-0.05, 1.05, 12),
        "phi": np.pi / 180 * np.arange(-105, 120, 30),
        "r2": np.linspace(-1.05, 1.05, 12),
    }
    widths = {"main_groups": 450, "sign": 400}
    for grp, cmap in cmaps.items():
        for col in ["amp", "ecc", "phi", "r2"]:
            df = base.copy()
            if grp == "sign":
                df = df[df.sign != 0]
                df["sign"] = df["sign"].astype(int)
                cmap_use = cmap.copy()
                cmap_use.index = cmap_use.index.astype(int)
            else:
                cmap_use = cmap
            fig = _hist_lines(df, col, grp, hist_bins[col], cmap_use,
                              height=400, width=widths[grp])
            fig.update_layout(xaxis_title=f"RF {col}", yaxis_title="frac. cell types",
                              yaxis_range=[-.01, 0.45])
            if col == "amp":
                fig.update_xaxes(type="log", range=[np.log10(1e-5), np.log10(1)])
            elif col == "phi":
                fig.update_xaxes(title="RF ori (deg)",
                                 tickvals=[-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2],
                                 ticktext=["-90", "-45", "0", "45", "90"])
            figs.append((fig, f"{dataset}_hist_rf_{col}_{grp}_types.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
_SIZE_DIFF_BINS = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
_SIZE_DIFF_TICKS = (_SIZE_DIFF_BINS[1:] + _SIZE_DIFF_BINS[:-1]) / 2
_SIZE_DIFF_LABELS = [-1, 0] + [int(2 ** i) for i in range(0, 8)]


# %%
def _bin_size_diff(df):
    d = df.copy()
    d["val"] = np.nan
    d.loc[d["size_diff"] < -0.5, "val"] = -2
    d.loc[(d["size_diff"] >= -0.5) & (d["size_diff"] <= 0.5), "val"] = -1
    d.loc[(d["size_diff"] >= 0.5) & (d["size_diff"] <= 1), "val"] = 0
    pos = d["size_diff"] >= 1
    d.loc[pos, "val"] = np.log2(d.loc[pos, "size_diff"])
    return d


# %%
def plot_rf_sizediff_internal(
    conn_edges: pd.DataFrame, colored_main_groups_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Histogram of `size_diff` weighted by weight for OL internal →
    (OL internal, OL output) edges. Needs only OL RFs (step 6a)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    d = _bin_size_diff(conn_edges[
        (conn_edges["main_groups_pre"] == "OL internal")
        & conn_edges["main_groups_post"].isin(["OL internal", "OL output"])
    ])
    fig = _hist_sum_lines(d, "val", "weight", "main_groups_post",
                          _SIZE_DIFF_BINS, colored_main_groups_df,
                          height=400, width=550)
    fig.update_layout(
        yaxis_title="frac. connections", xaxis_title="(post - pre)/pre ARF size",
        xaxis_tickvals=_SIZE_DIFF_TICKS, xaxis_tickangle=0,
        xaxis_ticktext=_SIZE_DIFF_LABELS, yaxis_range=[-0.01, 0.51],
        font=dict(family="arial", size=20),
    )
    _save(fig, save_dir, f"{dataset}_sizediff_internal_conn_post.pdf")
    return fig


# %%
def plot_rf_sizediff_output(
    conn_edges: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Histogram of `size_diff` weighted by weight for OL output →
    (right OL, CB) edges. Needs CB RFs (step 8) for the CB-post curve;
    otherwise only the right-OL line appears."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    d = _bin_size_diff(conn_edges[
        (conn_edges["main_groups_pre"] == "OL output")
        & conn_edges["region_post"].isin(["right OL", "CB"])
    ])
    if d.empty:
        return None
    fig = _hist_sum_lines(d, "val", "weight", "region_post",
                          _SIZE_DIFF_BINS, colored_region_df,
                          height=400, width=550)
    fig.update_layout(
        yaxis_title="frac. connections", xaxis_title="(post - pre)/pre ARF size",
        xaxis_tickvals=_SIZE_DIFF_TICKS, xaxis_tickangle=0,
        xaxis_ticktext=_SIZE_DIFF_LABELS, yaxis_range=[-0.01, 0.51],
        font=dict(family="arial", size=20),
    )
    _save(fig, save_dir, f"{dataset}_sizediff_output_conn_post.pdf")
    return fig


# %%
def plot_rf_connectivity_histograms(
    conn_edges: pd.DataFrame, colored_main_groups_df: pd.DataFrame,
    *, colored_region_df: pd.DataFrame | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Both size-diff histograms (internal + output) in one call. Pass
    `colored_region_df` to get the CB curve in the output panel."""
    figs = [plot_rf_sizediff_internal(conn_edges, colored_main_groups_df,
                                       dataset=dataset, save_dir=save_dir)]
    region_df = (
        colored_region_df if colored_region_df is not None
        else pd.DataFrame(
            [colored_main_groups_df.loc["OL output", "color"]],
            index=["right OL"], columns=["color"],
        )
    )
    out = plot_rf_sizediff_output(conn_edges, region_df,
                                   dataset=dataset, save_dir=save_dir)
    if out is not None:
        figs.append(out)
    return figs


# %%
def plot_rf_connectivity_boxes(
    conn_edges: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Size-diff binned (log2 after +2 shift) × layer-diff box plots, colored
    by main_groups_pre and by sign_pre. Only OL-internal + OL-output pres
    with aggregated weight > 5."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    df = conn_edges.copy()
    col_bins = np.arange(0, 10, 1, dtype=float) - .5
    col_labels = (col_bins[1:] + col_bins[:-1]) / 2
    df["col_bin_diff"] = pd.cut(
        np.log2(df["size_diff"].values + 2), bins=col_bins, labels=col_labels,
        include_lowest=True,
    ).astype(float)
    df = df.groupby(
        ["instance_pre", "instance_post", "main_groups_pre", "sign_pre"], dropna=False,
    ).agg({"layer_diff": "median", "col_bin_diff": "median", "weight": "median"}).reset_index()
    df = df[df["main_groups_pre"].isin(["OL internal", "OL output"]) & (df["weight"] > 5)]

    figs = []
    color_val = {"main_groups_pre": colored_main_groups_df, "sign_pre": colored_sign_df}
    for grp, cmap in color_val.items():
        sub = df.copy()
        if grp == "sign_pre":
            sub["sign_pre"] = sub["sign_pre"].astype(int)
            cmap = cmap.copy()
            cmap.index = cmap.index.astype(int)
        fig = _box_scatter(sub, "col_bin_diff", "layer_diff", grp, cmap,
                           height=400, width=500)
        fig.update_layout(
            xaxis_title="(post - pre)/pre RF size", yaxis_title="post - pre layer",
            xaxis_tickvals=col_labels,
            xaxis_ticktext=["%d" % (2 ** i - 2) for i in col_labels],
        )
        figs.append((fig, f"{dataset}_layerdiff_vs_sizediff_{grp}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %% [markdown]
# === 7. RF size experimental comparisons ===


# %%
def plot_rf_size_example_comparison(
    comparison_body: pd.DataFrame, *,
    star_instance: str, example_bid: int,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Scatter of direct-vs-propagated RF centre (x0, y0) and size per body for
    one star instance. Legacy cells 7-9."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "exp_comparison"
    figs = []
    sub = comparison_body[comparison_body["instance"] == star_instance].copy()
    if sub.empty:
        return figs
    sub["bodyId"] = sub["bodyId"].astype(str)
    # legacy: subtract 22 from y so y-range is centered around 0
    sub["y0_dir"] = sub["y0_dir"] - 22
    sub["y0_prop"] = sub["y0_prop"] - 22
    color_gray = pd.DataFrame(["lightgray"], columns=["color"], index=[star_instance])
    color_dark = pd.DataFrame(["darkgray"], columns=["color"], index=[star_instance])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-60, 60], y=[-60, 60], mode="lines",
                             line=dict(color="gray", dash="dash"),
                             showlegend=False, hoverinfo="skip"))
    fig2 = _scatter(sub, "y0_dir", "y0_prop", "instance", "bodyId",
                    color_gray, t_stars=[str(example_bid)],
                    height=350, width=350, marker="cross")
    fig1 = _scatter(sub, "x0_dir", "x0_prop", "instance", "bodyId",
                    color_dark, t_stars=[str(example_bid)],
                    height=350, width=350)
    for tr in fig1.data:
        fig.add_trace(tr)
    for tr in fig2.data:
        fig.add_trace(tr)
    fig.update_layout(fig1.layout)
    fig.update_layout(
        xaxis_title="direct RF x0 / y0", yaxis_title="propagated RF x0 / y0",
        xaxis_range=[-30, 30], yaxis_range=[-30, 30], height=350, width=350,
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    figs.append((fig, f"{dataset}_{star_instance}_rf_center_comp.pdf"))

    fig = _scatter(sub, "size_dir", "size_prop", "instance", "bodyId",
                   color_dark, t_stars=[str(example_bid)],
                   height=350, width=350)
    fig.add_trace(go.Scatter(x=[0, 20], y=[0, 20], mode="lines",
                             line=dict(color="gray", dash="dash"),
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        xaxis_title="direct ARF size", yaxis_title="indirect ARF size",
        xaxis_range=[-1, 20], yaxis_range=[-1, 20],
    )
    figs.append((fig, f"{dataset}_{star_instance}_rf_size_comp.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_rf_size_type_scatters(
    comparison_type: pd.DataFrame, colored_main_groups_df: pd.DataFrame,
    *, star_types: list[str] | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Per-instance direct-vs-propagated scatters for each RF param. Legacy
    cell 12."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "exp_comparison"
    stars = list(star_types) if star_types is not None else [s + "_R" for s in _RF_STAR_OUTPUT]
    sub = comparison_type[comparison_type["main_groups"].isin(["OL internal", "OL output"])].copy()
    figs = []
    for val in ["size", "n_col", "r2", "ecc", "phi"]:
        fig = _scatter(
            sub, f"{val}_dir", f"{val}_prop", "main_groups", "instance",
            colored_main_groups_df, stars,
        )
        fig.add_trace(go.Scatter(
            x=[-1000, 1000], y=[-1000, 1000], mode="lines",
            line=dict(color="gray", width=2, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.update_layout(
            yaxis_title=f"ARF {val}", xaxis_title=f"direct {val}",
            xaxis_title_standoff=0, yaxis_title_standoff=2,
        )
        if val == "size":
            fig.update_xaxes(type="log", tickformat="d", dtick=1,
                             range=[np.log10(0.6), np.log10(350)])
            fig.update_yaxes(type="log", tickformat="d", dtick=1,
                             range=[np.log10(0.6), np.log10(350)])
        elif val == "n_col":
            fig.update_xaxes(type="log", tickformat="d", dtick=1,
                             range=[np.log10(2), np.log10(1200)])
            fig.update_yaxes(type="log", tickformat="d", dtick=1,
                             range=[np.log10(2), np.log10(1200)])
        elif val == "r2":
            fig.update_xaxes(range=[-1.1, 1.1])
            fig.update_yaxes(range=[-1.1, 1.1])
        elif val == "ecc":
            fig.update_xaxes(range=[-0.1, 1.1])
            fig.update_yaxes(range=[-0.1, 1.1])
        elif val == "phi":
            ticks = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
            labels = ["-90", "-45", "0", "45", "90"]
            fig.update_xaxes(title="direct ori (deg)", tickvals=ticks, ticktext=labels)
            fig.update_yaxes(title="ARF ori (deg)", tickvals=ticks, ticktext=labels)
        figs.append((fig, f"{dataset}_type_{val}_dir_vs_prop.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def _mixed_color_df(colored_main_groups_df, colored_region_df):
    return pd.concat(
        (colored_main_groups_df.iloc[:-1], colored_region_df.iloc[2:3]),
        axis=0,
    )


# %%
def _add_2x_guide_lines(fig, lo: float, hi: float, *, color: str = "gray", width: float = 2):
    """Add three dashed guide lines: y=x, y=2x, y=x/2 over the log-range [lo, hi]."""
    for xs, ys in (
        ([lo, hi], [lo, hi]),
        ([lo, hi / 2], [2 * lo, hi]),
        ([2 * lo, hi], [lo, hi / 2]),
    ):
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=width, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))


# %%
def plot_rf_size_vs_experiment(
    comparison_type: pd.DataFrame, exp_sizes: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    col_to_deg_size: float = 25.0, col_to_deg_axis: float = 5.0,
):
    """Experimental vs predicted (propagated + direct) RF size scatter, plus
    sx/sy variants. Legacy cells 15-19.

    `col_to_deg_size`: degrees² per column (legacy default 25).
    `col_to_deg_axis`: degrees per column-radius (legacy default 5).
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "exp_comparison"
    mixed = _mixed_color_df(colored_main_groups_df, colored_region_df)
    instances_comp = [s + "_R" for s in exp_sizes["cell_type"].values]

    prop = comparison_type[[
        "instance", "cell_type_prop", "main_groups", "sign",
        "size_prop", "sx_prop", "sy_prop",
        "size_dir", "sx_dir", "sy_dir",
    ]].copy().rename(columns={"cell_type_prop": "cell_type"})
    sub = prop[prop["instance"].isin(instances_comp)].copy()
    sub = sub.merge(
        exp_sizes[["cell_type", "sx", "sy", "size"]].rename(
            columns={"size": "size_exp_deg", "sx": "sx_exp_deg", "sy": "sy_exp_deg"},
        ),
        on="cell_type", how="left",
    )
    sub["size_pred_deg"] = sub["size_prop"] * col_to_deg_size
    sub["sx_pred_deg"] = sub["sx_prop"] * col_to_deg_axis
    sub["sy_pred_deg"] = sub["sy_prop"] * col_to_deg_axis
    sub["size_dir_deg"] = sub["size_dir"] * col_to_deg_size
    sub["sx_dir_deg"] = sub["sx_dir"] * col_to_deg_axis
    sub["sy_dir_deg"] = sub["sy_dir"] * col_to_deg_axis

    from scipy.stats import pearsonr
    for val in ("size_pred_deg", "size_dir_deg"):
        nn = sub[["size_exp_deg", val]].dropna()
        if len(nn):
            corr, pval = pearsonr(nn["size_exp_deg"].values, nn[val].values)
            print(f"Pearson corr exp vs {val}: {corr:.3f} (p={pval:.2e}, n={len(nn)})")

    figs = []
    panels = [
        ("size_exp_deg", "size_pred_deg", "size_dir_deg", "RF size (deg²)",
         "exp_size_comp", (np.log10(5), np.log10(1000))),
        ("sx_exp_deg", "sx_pred_deg", "sx_dir_deg", "sx (deg)",
         "exp_sx_comp", (np.log10(0.7), np.log10(20))),
        ("sy_exp_deg", "sy_pred_deg", "sy_dir_deg", "sy (deg)",
         "exp_sy_comp", (np.log10(0.7), np.log10(20))),
    ]
    for x_col, y_prop, y_dir, label, stem, rng in panels:
        plot_df = sub.dropna(subset=[x_col, y_prop]).copy()
        if plot_df.empty:
            continue
        all_stars = list(plot_df["instance"].values)
        fig = _scatter(plot_df, x_col, y_prop, "main_groups", "instance",
                       mixed, all_stars, height=600, width=600, size=12)
        _add_2x_guide_lines(fig, 10 ** rng[0], 10 ** rng[1])
        for _, r in plot_df.iterrows():
            if pd.notna(r.get(y_dir)):
                fig.add_trace(go.Scatter(
                    x=[r[x_col], r[x_col]], y=[r[y_prop], r[y_dir]],
                    mode="lines", line=dict(color="black", width=1),
                    showlegend=False, hoverinfo="skip",
                ))
        dir_df = plot_df.dropna(subset=[y_dir])
        fig2 = _scatter(dir_df, x_col, y_dir, "main_groups", "instance",
                        mixed, list(dir_df["instance"].values),
                        height=600, width=600, marker="square", size=12,
                        show_labels=False)
        for tr in fig2.data:
            fig.add_trace(tr)
        fig.update_layout(
            xaxis_title=f"measured {label}", yaxis_title=f"predicted {label}",
            xaxis_title_standoff=0, yaxis_title_standoff=2,
        )
        fig.update_xaxes(type="log", tickformat="d", dtick=1, range=list(rng))
        fig.update_yaxes(type="log", tickformat="d", dtick=1, range=list(rng))
        figs.append((fig, f"{dataset}_{stem}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_rf_size_vs_experiment_cc(
    type_ol: pd.DataFrame, cc_sizes: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    col_to_deg_size: float = 25.0,
):
    """CurrierClandinin experimental vs propagated RF size (center vs ellipse).
    Legacy cell 23."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "exp_comparison"
    mixed = _mixed_color_df(colored_main_groups_df, colored_region_df)

    prop = type_ol[["instance", "cell_type", "main_groups", "sign", "size"]].copy()
    prop = pd.concat([
        prop,
        pd.DataFrame({
            "instance": ["L1_R", "L2_R", "L3_R"],
            "cell_type": ["L1", "L2", "L3"],
            "size": [1.0, 1.0, 1.0],
            "main_groups": ["visual input"] * 3,
            "sign": ["-1", "1", "1"],
        }),
    ], axis=0, ignore_index=True)
    prop["size_pred_deg"] = prop["size"] * col_to_deg_size

    sub = cc_sizes[["cell_type", "size_center", "size_ellipse"]].merge(
        prop, on="cell_type", how="left",
    )

    center_df = sub.dropna(subset=["size_center", "size_pred_deg"])
    fig = _scatter(center_df, "size_center", "size_pred_deg",
                   "main_groups", "instance", mixed,
                   list(center_df["instance"].values),
                   height=600, width=600, size=12)
    _add_2x_guide_lines(fig, 5, 1000)
    for _, r in sub.dropna(subset=["size_center", "size_ellipse", "size_pred_deg"]).iterrows():
        fig.add_trace(go.Scatter(
            x=[r["size_center"], r["size_ellipse"]],
            y=[r["size_pred_deg"], r["size_pred_deg"]],
            mode="lines", line=dict(color="black", width=0.5),
            showlegend=False, hoverinfo="skip",
        ))
    ellipse_df = sub.dropna(subset=["size_ellipse", "size_pred_deg"])
    fig2 = _scatter(ellipse_df, "size_ellipse", "size_pred_deg",
                    "main_groups", "instance", mixed,
                    list(ellipse_df["instance"].values),
                    height=600, width=600, marker="square", size=12,
                    show_labels=False)
    for tr in fig2.data:
        fig.add_trace(tr)
    fig.update_layout(
        xaxis_title="measured RF size (deg²) [CC]",
        yaxis_title="predicted ARF size (deg²)",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickvals=[10, 20, 50, 100, 200, 500],
                     range=[np.log10(15), np.log10(500)])
    fig.update_yaxes(type="log", tickvals=[10, 20, 50, 100, 200, 500],
                     range=[np.log10(15), np.log10(500)])
    _save(fig, save_dir, f"{dataset}_exp_size_comp_cc.pdf", html=True)
    return [(fig, f"{dataset}_exp_size_comp_cc.pdf")]


# %% [markdown]
# === 8. CB VIC ===


# %%
def plot_vic_cumsum(
    vic_type: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, vic_thre: float = 5e-4, dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """Cumulative VIC histogram per region with the CB threshold reference
    (legacy nb 8 cell 37)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    bins = 10 ** np.linspace(-6.5, -0.2, 31)
    fig = _hist_cumsum_lines_v(
        vic_type, "VIC", "region", bins, colored_region_df,
        height=400, width=300,
    )
    fig.add_hline(y=vic_thre, line_dash="dash", line_color="gray")
    fig.update_yaxes(type="log", range=[np.log10(2e-7), np.log10(0.8)])
    _save(fig, save_dir, f"{dataset}_vic_cumsum.pdf")
    return fig


# %%
def plot_vic_layer_histogram(
    vic_type: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, vic_thre: float = 5e-4, dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """Layer histogram of VCBN types (VIC > threshold) by region (legacy nb 8
    cell 39). Requires `vic_type` to have a `hitting_time` column."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    bins = np.arange(0.5 - 0.25 / 2, 7 + 0.25 / 2, 0.5 / 2)
    sub = vic_type[vic_type.VIC > vic_thre]
    fig = _hist_lines(
        sub, "hitting_time", "region", bins, colored_region_df,
        height=230, width=450,
    )
    fig.update_xaxes(range=[0.5, 6], tickvals=[1, 2, 3, 4, 5, 6])
    fig.update_yaxes(tickvals=[0, 0.3])
    fig.update_layout(xaxis_title="layer", yaxis_title="frac.")
    fig.update_layout(xaxis_title_standoff=0, yaxis_title_standoff=2)
    _save(fig, save_dir, f"{dataset}_vic_layer_hist.pdf")
    return fig


# %%
def plot_vic_vs_layer_scatter(
    vic_type: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, vic_thre: float = 5e-4, dataset: str = DATASET,
    save_dir: Path | None = None, highlight: list[str] | None = None,
    matched_instances: list[str] | None = None,
    show_highlight_labels: bool = True,
):
    """Scatter of VIC vs hitting_time (layer), one point per instance, colored
    by region. Optional `highlight` and `matched_instances` lists for labeled
    callouts (legacy nb 8 cells 41-42)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    highlight = list(highlight) if highlight is not None else []
    matched_instances = list(matched_instances) if matched_instances is not None else []

    fig = _scatter(
        vic_type, "hitting_time", "VIC", "region", "instance",
        colored_region_df, highlight, show_labels=show_highlight_labels,
    )
    fig.add_hline(y=vic_thre, line_dash="dash", line_color="gray")
    if matched_instances:
        matched = vic_type[vic_type["instance"].isin(matched_instances)]
        fig2 = _scatter(
            matched, "hitting_time", "VIC", "region", "instance",
            colored_region_df, matched_instances, show_labels=show_highlight_labels,
        )
        for tr in fig2.data:
            fig.add_trace(tr)
    fig.update_yaxes(type="log", range=[np.log10(2e-7), np.log10(0.8)])
    fig.update_xaxes(range=[0.5, 6], tickvals=[1, 2, 3, 4, 5, 6])
    fig.update_layout(xaxis_title="layer")
    fig.update_layout(xaxis_title_standoff=0, yaxis_title_standoff=2)
    _save(fig, save_dir, f"{dataset}_vic_vs_layer.pdf", html=True, rasterize=True)
    return fig


# %% [markdown]
# === 9. LR CB clustering ===


# %%
# Legacy `load_cb_cluster_types(side=...)`: full-suffix names with a
# deliberate L/R mix per side. Keep the order exactly as in legacy so the
# example stacked bar keeps the same left-to-right layout.
_CB_CLUSTER_TYPES = {
    "r": [
        "pC1_2a_R", "DNg02_a_R", "IbSpsP_0", "TuBu08_R", "SMP228_R",
        "ER4d_R", "PVLP112_R", "PVLP113_R", "DNp02_R", "pC1_5b_R",
    ],
    "l": [
        "pC1_2a_L", "DNg02_a_L", "IbSpsP_0", "TuBu08_L", "SMP228_L",
        "ER4d_L", "PVLP112_L", "PVLP113_L", "DNp02_L", "pC1_5b_L",
    ],
}


# %%
def _cb_cluster_labels(n_clu: int, side_char: str = "r") -> list[str]:
    prefix = "cc'" if side_char.lower().startswith("l") else "cc"
    return [f"{prefix}{i}" for i in range(1, n_clu + 1)]


# %%
def _cb_cluster_side(cluster_df: pd.DataFrame, side_str: str = "") -> str:
    if side_str.startswith("_left"):
        return "l"
    if "instance" in cluster_df.columns:
        inst = cluster_df["instance"].astype(str)
        if len(inst) and inst.str.endswith("_L").all():
            return "l"
    return "r"


# %%
def plot_cb_directedness_histograms(
    type_dir_cb: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Per-direction (ff/fb/la) fraction-of-types histogram for VCBN pre-neurons.

    Legacy nb 8 cell 73. `type_dir_cb` comes from `pre.get_cb_type_directedness()`;
    `main_groups_pre` is collapsed to `'nonOL'` there, so all types share one
    colour. We build a one-row colour df that maps `'nonOL'` to the CB region
    colour.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    cmap = pd.DataFrame({"color": [colored_region_df.loc["CB", "color"]]}, index=["nonOL"])
    figs = []
    ff_bins = np.arange(-0.1, 1.2, 0.2)
    for direct in ["ff", "fb", "la"]:
        fig = _hist_lines(type_dir_cb, f"frac_{direct}", "main_groups_pre", ff_bins, cmap)
        fig.update_layout(
            xaxis_title=f"frac_{direct}", xaxis_range=[-.05, 1.05],
            yaxis_title="frac. cell types", yaxis_range=[-.01, 1.01],
            xaxis_title_standoff=0, yaxis_title_standoff=2,
        )
        figs.append((fig, f"{dataset}_{direct}_frac_hist_all.pdf"))
    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_cb_directedness_triangle(
    type_dir_cb: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, cluster_types=None, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Ternary directedness triangle (la/fb/ff) for VCBN pre-neurons.

    Legacy nb 8 cell 74. Applies the legacy eps=0.05 smoothing before plotting
    and highlights `cluster_types` (defaults to `_CB_CLUSTER_TYPES['r']`).
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    stars = list(cluster_types) if cluster_types is not None else _CB_CLUSTER_TYPES["r"]
    cmap = {"nonOL": colored_region_df.loc["CB", "color"]}
    d = type_dir_cb.copy()
    eps = 5e-2
    d[["frac_la", "frac_fb", "frac_ff"]] = d[["frac_la", "frac_fb", "frac_ff"]] * (1 - eps) + eps / 3
    fig = _directedness_triangle(d, "main_groups_pre", "instance_pre", cmap, stars)
    _save(fig, save_dir, f"{dataset}_examples_0.1thre_dir_triangle.pdf", html=True)
    return fig


# %%
def plot_cb_clusters_example_features(
    feature_df: pd.DataFrame, in_instances, cluster_types,
    colored_seed_df: pd.DataFrame, *,
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
):
    """Stacked bar of normalised CB feature vector for example output types
    (legacy nb 8 cell 56). `cluster_types` are full instance names (with
    side suffix) from `_CB_CLUSTER_TYPES` — L/R mix allowed."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    keep = [t for t in cluster_types if t in feature_df.index]
    sub = feature_df.loc[keep, in_instances]
    plot_df = sub / sub.sum(1).values[:, np.newaxis]
    plot_df.columns = [s[:-2] for s in plot_df.columns]
    plot_df.index.name = None
    plot_df = plot_df.stack(0).reset_index(name="effective weight").rename(
        columns={"level_0": "out type", "level_1": "in type"}
    )
    fig = _stacked_bars(
        plot_df, "out type", "effective weight", "in type",
        colored_seed_df["color"].to_dict(),
        xrot=-45, height=450, width=550,
    )
    fig.update_layout(xaxis_title_standoff=0, yaxis_title_standoff=2)
    _save(fig, save_dir, f"{dataset}{side_str}_cb_in_out_examples.pdf")
    return fig


# %%
def plot_cb_clusters_pathway_features(
    feature_df: pd.DataFrame, in_instances, cluster_df: pd.DataFrame,
    cluster_num: pd.Series, colored_seed_df: pd.DataFrame, *,
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
):
    """Stacked bar of per-cluster mean normalised CB feature vector
    (legacy nb 8 cell 57). Labels right clusters as cc1..ccN and left clusters
    as cc'1..cc'N, with member count."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    sub = feature_df.loc[:, in_instances] / feature_df.loc[:, in_instances].sum(1).values[:, np.newaxis]
    df = sub.reset_index(names="instance").merge(cluster_df, on="instance")
    df = df.groupby("cluster")[in_instances].mean()
    df.columns = [s[:-2] for s in df.columns]
    df = df.stack(0).reset_index(name="mean effective weight").rename(
        columns={"level_0": "cluster", "level_1": "in type"}
    )
    fig = _stacked_bars(
        df, "cluster", "mean effective weight", "in type",
        colored_seed_df["color"].to_dict(), height=400, width=495,
    )
    n_clu = len(cluster_num)
    cluster_labels = _cb_cluster_labels(n_clu, _cb_cluster_side(cluster_df, side_str))
    fig.update_layout(
        xaxis_tickvals=np.arange(1, n_clu + 1),
        xaxis_ticktext=[
            f"{cluster_labels[i - 1]} ({cluster_num[i]})"
            for i in range(1, n_clu + 1)
        ],
        xaxis_tickangle=-45,
        xaxis_title_standoff=0, yaxis_title_standoff=2,
        font=dict(family="arial", size=16),
    )
    _save(fig, save_dir, f"{dataset}{side_str}_cb_in_out_pathways.pdf")
    return fig


# %%
def plot_ol_cb_cluster_connectivity(
    conn: pd.DataFrame, *, dataset: str = DATASET,
    save_dir: Path | None = None, anno_min_frac: float = 0.05,
):
    """Two heatmaps: OL-cluster × CB-cluster connectivity, input-normalised
    (each CB column sums to 1) and output-normalised (each OL row sums to 1).
    Annotations show raw total weight (K/M) where normalised ≥ `anno_min_frac`.
    Legacy nb 8 cells 70-71."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"

    def _format_total(v):
        if v >= 1e6:
            return f"{v / 1e6:.0f}M"
        if v >= 1e3:
            return f"{v / 1e3:.0f}K"
        return f"{v:.0f}"

    def _one(norm_df, total_df, title, stem):
        anno = np.empty(total_df.shape, dtype=object)
        for j in range(total_df.shape[1]):
            for i in range(total_df.shape[0]):
                anno[i, j] = (
                    _format_total(total_df.values[i, j])
                    if norm_df.values[i, j] > anno_min_frac else ""
                )
        fig = go.Figure(data=go.Heatmap(
            z=norm_df.values, x=list(norm_df.columns), y=list(norm_df.index),
            colorscale="Greys", showscale=True, zmin=0, zmax=1,
            text=anno, texttemplate="%{text}", textfont=dict(size=18),
        ))
        fig.update_layout(
            font=dict(family="arial", size=18),
            title=title, xaxis_title="", yaxis_title="",
            xaxis_tickvals=list(norm_df.columns),
            xaxis_ticktext=[f"cc{c}" for c in norm_df.columns],
            yaxis_tickvals=list(norm_df.index),
            yaxis_ticktext=[f"oc{c}" for c in norm_df.index],
            paper_bgcolor="white", plot_bgcolor="white",
            height=550, width=550,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black",
                         mirror=True, title_standoff=0)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black",
                         mirror=True, title_standoff=2, autorange="reversed")
        _save(fig, save_dir, f"{dataset}_{stem}.pdf")
        return fig

    total = conn.copy()
    col_sum = np.abs(total).sum(axis=0).replace(0, 1)
    row_sum = np.abs(total).sum(axis=1).replace(0, 1)
    norm_in = total.div(col_sum, axis=1)
    norm_out = total.div(row_sum, axis=0)

    fig_in = _one(norm_in, total,
                  "input-normalised connectivity from OL output clusters to CB clusters",
                  "cb_clusters_in")
    fig_out = _one(norm_out, total,
                   "output-normalised connectivity from OL output clusters to CB clusters",
                   "cb_clusters_out")
    return fig_in, fig_out


# %%
_CB_ROI_ORDER = [
    "PVLP(R)", "PVLP(L)", "SLP(R)", "IB", "CRE(R)",
    "AVLP(R)", "AVLP(L)", "EB", "IPS(R)", "SIP(R)",
    "PLP(R)", "PLP(L)", "SMP(R)", "ICL(R)", "VES(R)",
    "SPS(R)", "SPS(L)", "GNG", "SCL(R)", "WED(R)",
    "LAL(R)", "LAL(L)", "AOTU(R)", "PB", "BU(R)",
]


# %%
def _fmt_km(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.0f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}"
    return f"{int(v)}"


# %%
def plot_cb_clusters_rois(
    tbar_counts: pd.DataFrame, *,
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
    roi_order: list[str] | None = None, anno_min: float = 0.1,
):
    """CB-cluster × ROI normalised-tbar heatmap (legacy nb 8 c 105)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    roi_order = roi_order if roi_order is not None else _CB_ROI_ORDER
    roi_keep = [r for r in roi_order if r in tbar_counts.columns]
    m = tbar_counts[roi_keep]
    total_all = tbar_counts.values.sum()
    total_shown = m.values.sum()
    frac_shown = total_shown / total_all if total_all > 0 else 0.0
    m_norm = m.div(m.sum(0), axis=1).fillna(0)

    anno = np.where(
        m_norm.values > anno_min,
        np.vectorize(_fmt_km)(m.values.astype(float)), "",
    )

    fig = go.Figure(data=go.Heatmap(
        z=m_norm.values, x=list(m_norm.columns), y=list(m_norm.index),
        colorscale="Greys", showscale=True, zmin=0, zmax=1,
        text=anno, texttemplate="%{text}", textfont=dict(size=20),
    ))
    fig.update_layout(
        font=dict(family="arial", size=21),
        title=f"frac. synapses per cluster (showing {frac_shown:.0%} of total)",
        xaxis_title="", yaxis_title="",
        xaxis_tickvals=list(range(len(m_norm.columns))),
        xaxis_ticktext=list(m_norm.columns),
        yaxis_tickvals=list(range(1, m_norm.shape[0] + 1)),
        yaxis_ticktext=[f"cc{i}" for i in m_norm.index],
        paper_bgcolor="white", plot_bgcolor="white",
        height=440, width=1188,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black",
                     mirror=True, tickangle=-90, title_standoff=0)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black",
                     mirror=True, autorange="reversed", title_standoff=2)
    _save(fig, save_dir, f"{dataset}{side_str}_cb_clusters_rois.pdf")
    return fig


# %%
def plot_cb_participation_n_clu_out(
    type_part_cb: pd.DataFrame, colored_main_groups_df: pd.DataFrame, *,
    side_char: str = SIDE_CHAR, dataset: str = DATASET,
    side_str: str = "", save_dir: Path | None = None,
):
    """Histogram of n_clusters (CB clusters with pi > 1/N) for OL-output
    cell types. Legacy nb 8 c 88, filename `…_in_out_n_clu_participation_out.pdf`."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    side_word = _SIDE_WORD[side_char] if side_char in _SIDE_WORD else "right"
    sub = type_part_cb[
        type_part_cb["region"].isin([f"{side_word} OL"])
        & (type_part_cb["main_groups"] == "OL output")
    ]
    cluster_names = [c for c in type_part_cb.columns if bool(re.match(r"^c\d+$", str(c)))]
    n_clu = len(cluster_names)
    bins = np.linspace(0, n_clu + 1, n_clu + 2) - 0.5
    fig = _hist_lines(sub, "n_clusters", "main_groups", bins, colored_main_groups_df)
    fig.update_layout(
        xaxis_title=f"no. clusters with pi>1/{n_clu}",
        xaxis_range=[-.1, n_clu + 0.1], xaxis_tickvals=np.arange(n_clu + 1),
        yaxis_title="frac. cell types", yaxis_range=[-.01, 0.61],
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    _save(fig, save_dir, f"{dataset}{side_str}_in_out_n_clu_participation_out.pdf")
    return fig


# %%
def plot_cb_clusters_graph(
    type_part_cb: pd.DataFrame, ol_cluster_df: pd.DataFrame,
    colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, side_str: str = "",
    save_dir: Path | None = None, edge_color: str = "lightgray",
    ol_highlight: list[str] | None = None,
):
    """igraph network of OL-output types + CB clusters, edges are
    participation ≥ 1/N. Legacy nb 8 c 99."""
    import igraph as ig
    import matplotlib.pyplot as plt
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    save_dir.mkdir(parents=True, exist_ok=True)

    cluster_names_c = [c for c in type_part_cb.columns if bool(re.match(r"^c\d+$", str(c)))]
    cluster_names_no_c = [s[1:] for s in cluster_names_c]
    n_clu = len(cluster_names_c)

    base = type_part_cb[type_part_cb["main_groups"] == "OL output"].copy()
    long = base.melt(
        id_vars=["instance"], value_vars=cluster_names_c,
        var_name="post", value_name="weight",
    ).rename(columns={"instance": "pre"})
    long = long[long["weight"] >= 1 / n_clu].copy()
    long["pre"] = [s.split("_")[0] for s in long["pre"]]
    long["post"] = [s[1:] for s in long["post"]]

    nodes = list(set(long["pre"]).union(long["post"]))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_to_idx[s], node_to_idx[t]) for s, t in zip(long["pre"], long["post"])]

    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    g.add_edges(edges)
    g.vs["name"] = nodes
    g.es["weight"] = long["weight"].tolist()

    # Color: CB cluster nodes share the CB region gray; OL-output cell types
    # keep the per-cluster blue gradient.
    cb_gray = colored_region_df.loc["CB", "color"]
    cluster_to_color = {str(i + 1): cb_gray for i in range(n_clu)}

    ol_color = ol_cluster_df.copy()
    ol_color["cell_type"] = [s.split("_")[0] for s in ol_color["instance"]]
    ol_blues = [
        "#041f4a", "#08306b", "#08519c", "#2171b5", "#4292c6",
        "#6baed6", "#9ecae1", "#c6dbef", "#deebf7",
    ]
    n_ol = int(ol_color["cluster"].max())
    ol_color["color"] = [ol_blues[(c - 1) % len(ol_blues)] for c in ol_color["cluster"]]
    pre_color = ol_color.set_index("cell_type")["color"].to_dict()
    pre_color.update(cluster_to_color)
    cb_fallback = colored_region_df.loc["CB", "color"]
    vertex_colors = [pre_color.get(v["name"], cb_fallback) for v in g.vs]

    ol_types = ol_highlight if ol_highlight is not None else _CLUSTER_TYPES
    highlight = set(cluster_names_no_c) | set(ol_types)
    vertex_label = [s if s in highlight else "" for s in g.vs["name"]]
    vertex_frame_color = ["black" if v["name"] in highlight else "white" for v in g.vs]
    vertex_size = [25 if v["name"][0].isdigit() else 15 for v in g.vs]

    layout = g.layout("kk")
    fig, ax = plt.subplots(figsize=(8, 6))
    ig.plot(
        g, target=ax, layout=layout,
        vertex_label=vertex_label, vertex_label_size=14,
        vertex_color=vertex_colors, vertex_frame_color=vertex_frame_color,
        vertex_frame_width=1, vertex_size=vertex_size,
        edge_color=edge_color, edge_arrow_size=0.0,
        edge_width=[0.1 + 3 * w for w in g.es["weight"]],
    )
    ax.axis("off")
    out = save_dir / f"{dataset}{side_str}_graph_cb_pathways_v1.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _save_graph_html(
        g, layout, vertex_colors, vertex_size,
        HTML_FIG_DIR / out.with_suffix(".html").name,
    )
    return out


# %%
def plot_out_cb_highres_sub(
    tb_clu_df: pd.DataFrame, sizes: pd.Series, *,
    thre_row: float = 1.0, thre_col: float = 0.8, thre_row2: float = 0.5,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Subset of OL-output × CB-type high-res effective-weight heatmap,
    filtered by row/col/row2 score thresholds. Legacy nb 10 cells 44-53."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    out_score = tb_clu_df.sum(1)
    ol_major = out_score[out_score > thre_row].index
    cb_score = tb_clu_df.loc[ol_major].sum(0)
    cb_major = cb_score[cb_score > thre_col].index
    out_score2 = tb_clu_df.loc[:, cb_major].sum(1)
    ol_major2 = out_score2[out_score2 > thre_row2].index
    sub = tb_clu_df.loc[ol_major2, cb_major]

    xtt = [s.split("_")[0] + f" ({int(sizes.get(s, 0))})" for s in sub.columns]
    ytt = [s.split("_")[0] + f" ({int(sizes.get(s, 0))})" for s in sub.index]
    fig = go.Figure(data=go.Heatmap(
        z=sub.values, x=list(sub.columns), y=list(sub.index),
        colorscale="Greys", showscale=True, zmin=0, zmax=1,
    ))
    fig.update_layout(
        font=dict(family="arial", size=16),
        title="rel effective weight from OL output to high-res. CB cell types (major only)",
        xaxis_title="high-resolution VCBN types (size in col)",
        yaxis_title="high-resolution VPN types (size in col)",
        xaxis_tickvals=list(range(sub.shape[1])), xaxis_ticktext=xtt,
        yaxis_tickvals=list(range(sub.shape[0])), yaxis_ticktext=ytt,
        paper_bgcolor="white", plot_bgcolor="white",
        height=600, width=1500,
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black",
                     mirror=True, tickangle=-45, title_standoff=0)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black",
                     mirror=True, autorange="reversed", title_standoff=2)
    _save(fig, save_dir, f"{dataset}_out_cb_highres_in_sub.pdf")
    return fig


# %%
def plot_rf_example_pos_par(
    meta_ol: pd.DataFrame, stepsn, raw_ol: pd.DataFrame, bodyId: int,
    in_instances: list[str], *,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Hex heatmap for a single bodyId with Gaussian ellipse overlay
    (legacy nb 6 c 20, `…_rf_pos_par.pdf`)."""
    from connectome_interpreter.external_map import hex_heatmap
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    row = raw_ol[raw_ol["bodyId"] == bodyId]
    if row.empty:
        return None
    params_single = row.iloc[[0]]

    inidx = meta_ol[meta_ol.instance.isin(in_instances)].idx.values
    outidx = meta_ol[meta_ol.bodyId == bodyId].idx.values
    from utils.ol_rf import compute_rf
    df = compute_rf(meta_ol, stepsn, inidx, outidx)
    if df.empty:
        return None
    plot_df = df.set_index("coords")[["effective weight"]]
    cmax = float(np.abs(plot_df["effective weight"]).max())

    fig = hex_heatmap(
        plot_df, custom_colorscale="Picnic",
        global_min=-1, global_max=1, dataset="mcns_right",
    )
    _add_gaussian_ellipses(fig, params_single, example_bid=bodyId, y_scale=np.sqrt(3))
    _save(fig, save_dir, f"{dataset}_example_{bodyId}_rf_pos_par.pdf")
    return fig


# %%
def plot_ol_layers_hit_diff(
    roi_adj: pd.DataFrame, rois: list[str], labels: list[str], *,
    filename_stem: str, dataset: str = DATASET, save_dir: Path | None = None,
    height: int = 800, width: int = 500,
):
    """Per-ROI distribution of connection layer differences (post - pre).

    For each anatomical OL ROI in `rois`, takes every (pre, post) connection
    with at least one synapse in that ROI, expands by `weight` so each
    individual synapse is a sample, and draws a horizontal violin of
    `hit_diff`. Y-axis labels are `labels` (e.g. M1..AME or LO1..LOP4),
    matching SI 1i. `roi_adj` comes from `get_ol_roi_adjacency`.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    sub = roi_adj[roi_adj["roi"].isin(rois)]
    sub = sub.loc[sub.index.repeat(sub["weight"])][["roi", "hit_diff"]]
    sub["roi"] = pd.Categorical(sub["roi"], categories=rois, ordered=True)
    sub = sub.sort_values("roi")
    # Downsample to avoid exceeding kaleido's 128 MB IPC message limit.
    _MAX_POINTS = 500_000
    if len(sub) > _MAX_POINTS:
        sub = sub.groupby("roi", observed=True, group_keys=False).apply(
            lambda g: g.sample(
                n=max(1, round(_MAX_POINTS * len(g) / len(sub))),
                random_state=0,
            )
        )
    fig = _violin_median(
        sub, "hit_diff", "roi", labels=labels, height=height, width=width,
        xaxis_title="post - pre layer", xaxis_tickvals=[-3, -2, -1, 0, 1, 2, 3],
        xaxis_range=[-3.5, 3.5],
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.write_image(save_dir / f"{dataset}_{filename_stem}_diff.png", scale=2)
    return fig


# %%
def plot_cb_layer_hist_left(
    layer_lr_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, *,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Horizontal histogram of left-side hitting time for OL output + CB
    instances (legacy nb 9 cell 21)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    cmap = colored_region_df.copy()
    cmap.loc["OL output", "color"] = colored_main_groups_df.loc["OL output", "color"]
    df = layer_lr_df.copy()
    id_col = "instance" if "instance" in df.columns else "cell_type"
    df = df[~df[id_col].str.contains("_unclear", regex=False)]
    ol = df[
        df["region"].isin(["left OL", "right OL"])
        & (df["main_groups"] == "OL output")
    ].copy()
    ol["region"] = "OL output"
    cb = df[df["region"] == "CB"]
    plot_df = pd.concat([ol, cb], axis=0, ignore_index=True)

    bins = np.arange(0.5 - 0.25 / 2, 7 + 0.25 / 2, 0.5 / 2)
    fig = _hist_lines_v(
        plot_df, "hitting_time_l", "region", bins, cmap,
        height=400, width=400,
    )
    fig.update_yaxes(range=[0.8, 6.5], tickvals=[1, 2, 3, 4, 5, 6])
    _save(fig, save_dir, "layer_hist_left.pdf")
    return fig


# %%
def plot_cb_clusters_size_vs_cluster(
    cb_cluster_df: pd.DataFrame, cb_type_df: pd.DataFrame,
    colored_region_df: pd.DataFrame, *,
    dataset: str = DATASET, save_dir: Path | None = None, jitter: float = 0.2,
    highlight: list[str] | None = None,
):
    """Scatter ARF size × cluster for CB types (legacy nb 10 cell 34,
    `examples_size_vs_cluster_v2`).

    `cb_type_df` must have `instance` + `size` columns (from `get_rf_type_cb`)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    n_clu = int(cb_cluster_df["cluster"].max())
    cluster_names = _cb_cluster_labels(n_clu, _cb_cluster_side(cb_cluster_df))
    stars = list(highlight) if highlight is not None else _CB_CLUSTER_TYPES.get("r", [])

    merged = cb_cluster_df.merge(
        cb_type_df[["instance", "size", "region"]], on="instance", how="inner",
    ).copy()
    rng = np.random.default_rng(0)
    merged["y"] = merged["cluster"].astype(float) + rng.uniform(-jitter, jitter, size=len(merged))

    fig = _scatter(
        merged, "size", "y", "region", "instance",
        colored_region_df, stars, height=400, width=500,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickvals=np.arange(1, n_clu + 1), ticktext=cluster_names,
    )
    fig.update_xaxes(type="log", tickformat="d",
                     tickvals=[5, 10, 50, 100], range=[np.log10(4.5), np.log10(250)])
    fig.update_layout(yaxis_title="cluster", xaxis_title="ARF size",
                      xaxis_title_standoff=0, yaxis_title_standoff=2)
    _save(fig, save_dir, f"{dataset}_examples_size_vs_cluster_v2.pdf", html=True)
    return fig


# %%
def plot_cb_clusters_scatter(
    cb_cluster_df: pd.DataFrame, layer_lr: pd.DataFrame,
    colored_region_df: pd.DataFrame, *,
    dataset: str = DATASET, side_str: str = "_cb",
    save_dir: Path | None = None, jitter: float = 0.2,
    highlight: list[str] | None = None,
):
    """Scatter layer (hitting_time) × cluster for CB instances, colored by
    region. Mirrors the OL `plot_clusters_scatter` layer-vs-cluster panel.
    `layer_lr` supplies per-instance `hitting_time_r` (right side layer)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    n_clu = int(cb_cluster_df["cluster"].max())
    cluster_names = _cb_cluster_labels(n_clu, _cb_cluster_side(cb_cluster_df, side_str))
    stars = list(highlight) if highlight is not None else _CB_CLUSTER_TYPES.get("r", [])

    base = cb_cluster_df.merge(
        layer_lr[["instance", "hitting_time_r", "region"]],
        on="instance", how="left",
    ).copy()
    base = base.dropna(subset=["hitting_time_r"])
    rng = np.random.default_rng(0)
    base["y"] = base["cluster"].astype(float) + rng.uniform(-jitter, jitter, size=len(base))

    fig = _scatter(
        base, "hitting_time_r", "y", "region", "instance",
        colored_region_df, stars, height=300, width=400,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickvals=np.arange(1, n_clu + 1), ticktext=cluster_names,
    )
    fig.update_xaxes(range=[1.5, 5.5])
    fig.update_layout(
        xaxis_title="layer", yaxis_title="cluster",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
        font=dict(family="arial", size=20),
    )
    _save(fig, save_dir, f"{dataset}{side_str}_examples_layer_vs_cluster.pdf", html=True)
    return fig


# %%
def plot_cb_clusters_intra_connectivity(
    intra: dict, *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """CB-cluster × CB-cluster connectivity heatmap (output-normalised).
    Writes `{dataset}_cb_outrel_conn_matrix_clusters.pdf` in `CB_clustering/`
    — `cb_` prefix distinguishes it from the OL analogue in `pathways/`.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    n = intra["n_clu"]
    tick_labels = _cb_cluster_labels(n, "r")
    fig = _heatmatrix(intra["matrix"], tick_labels, cmap="Greys", height=600, width=600)
    fig.update_layout(
        title=f"all weights: {intra['total_weight']: .0f}",
        xaxis_title="post", yaxis_title="pre", xaxis_tickangle=0,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(
        colorbar=dict(tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"]),
        zmin=0, zmax=1,
    )
    _save(fig, save_dir, f"{dataset}_cb_outrel_conn_matrix_clusters.pdf")
    return fig


# %%
def plot_cb_lr_clustering_sweep(
    sweep_df: pd.DataFrame, n0: int, *,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Line panels for the CB L/R clustering threshold sweep.

    `n0` is the reference cluster count (from the chosen `frac_thre` in
    section 5). Panels: n_R vs frac, n_L vs n_R, ARI_LR vs n_R, ARI_fromR vs
    n_R, ARI_fromR vs ARI_LR.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    matches = np.where(sweep_df["n_clu_R"].values == n0)[0]
    idx0 = int(matches[0]) if len(matches) else int(np.argmin(
        np.abs(sweep_df["n_clu_R"].values - n0)
    ))
    figs = []

    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["frac"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.update_layout(xaxis_title="No. R clusters", yaxis_title="Clustering threshold")
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0, 2])
    fig.update_yaxes(type="log", tickvals=[0.05, 0.1, 0.5, 1], range=[-1.7, 0.1])
    figs.append((fig, f"{dataset}_cb_no_thre.pdf"))

    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["n_clu_L"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.add_trace(go.Scatter(x=[1, 300], y=[1, 300], mode="lines",
                             line=dict(color="gray", dash="dash"), showlegend=False))
    fig.update_layout(
        xaxis_title="no. R clusters", yaxis_title="no. L clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    figs.append((fig, f"{dataset}_cb_LR_no_clusters.pdf"))

    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["ari_LR"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.update_layout(
        xaxis_title="no. R clusters", yaxis_title="ARI between L and R clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_cb_LR_ARI.pdf"))

    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["ari_fromR"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0])
    fig.update_layout(
        xaxis_title="no. R clusters", yaxis_title="ARI between L and L from R clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_cb_LR_fromR_ARI.pdf"))

    df = pd.DataFrame({"x": sweep_df["ari_fromR"].values, "y": sweep_df["ari_LR"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=500)
    fig.add_trace(go.Scatter(x=[0.01, 1], y=[0.01, 1], mode="lines",
                             line=dict(color="gray", dash="dash"), showlegend=False))
    fig.update_layout(
        yaxis_title="ARI between L and R clusters",
        xaxis_title="ARI between L and L from R clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_cb_LR_ARI_comparison.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_cb_lr_cluster_confusion(
    match: dict, *,
    dataset: str = DATASET, save_dir: Path | None = None, anno_min: int = 10,
    filename_stem: str = "cb_left_to_right_confusion",
    xaxis_title: str = "right cluster", yaxis_title: str = "left cluster",
):
    """Heatmap of the CB L↔R cluster confusion matrix."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    conf = match["confusion"]
    x_tick_labels = match.get("x_tick_labels")
    y_tick_labels = match.get("y_tick_labels")
    if x_tick_labels is None or y_tick_labels is None:
        tick_labels = match.get("tick_labels")
        if tick_labels is not None:
            x_tick_labels = list(tick_labels)[:conf.shape[1]]
            y_tick_labels = list(tick_labels)[:conf.shape[0]]
        else:
            x_tick_labels = _cb_cluster_labels(conf.shape[1], "r")
            y_tick_labels = _cb_cluster_labels(conf.shape[0], "l")
    if len(x_tick_labels) != conf.shape[1]:
        x_tick_labels = _cb_cluster_labels(conf.shape[1], "r")
    if len(y_tick_labels) != conf.shape[0]:
        y_tick_labels = _cb_cluster_labels(conf.shape[0], "l")
    anno = np.where(conf >= anno_min, conf.astype(object), "")
    fig = _heatmatrix(
        conf, x_tick_labels, y_tick_labels=y_tick_labels,
        anno=anno, width=550,
    )
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    _save(fig, save_dir, f"{dataset}_{filename_stem}.pdf")
    return fig


# %%
def _plot_layer_lr_scatter(
    df: pd.DataFrame, id_col: str, colored_region_df: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, *,
    highlight: list[str] | None = None,
):
    """Shared scatter builder for right-layer vs left-layer. Filters to OL
    output + CB rows, recolors OL output with `colored_main_groups_df`."""
    cmap = colored_region_df.copy()
    cmap.loc["OL output", "color"] = colored_main_groups_df.loc["OL output", "color"]
    df = df[~df[id_col].str.contains("_unclear", regex=False)]
    ol = df[
        df["region"].isin(["left OL", "right OL"])
        & (df["main_groups"] == "OL output")
    ].copy()
    ol["region"] = "OL output"
    cb = df[df["region"] == "CB"]
    plot_df = pd.concat([ol, cb], axis=0, ignore_index=True)
    stars = list(highlight) if highlight else []
    fig = _scatter(
        plot_df, "hitting_time_r", "hitting_time_l", "region", id_col,
        cmap, stars,
    )
    fig.add_trace(go.Scatter(
        x=[0.8, 7.5], y=[0.8, 7.5], mode="lines",
        line=dict(color="gray", dash="dash"), showlegend=False,
    ))
    fig.update_layout(
        xaxis_title="right layer", yaxis_title="left layer",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    return fig


# %%
def plot_cb_layer_lr_homologue(
    layer_lr_homologue: pd.DataFrame, colored_region_df: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, *,
    dataset: str = DATASET, save_dir: Path | None = None,
    highlight: list[str] | None = None,
):
    """Cell-type-matched right-layer vs left-layer scatter (legacy nb 9
    `layer_lr_comp` panel). Takes output of `get_cb_layer_lr_homologue`."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    fig = _plot_layer_lr_scatter(
        layer_lr_homologue, "cell_type", colored_region_df,
        colored_main_groups_df, highlight=highlight,
    )
    fig.update_xaxes(range=[0.8, 6.5], tickvals=[1, 2, 3, 4, 5, 6])
    fig.update_yaxes(range=[0.8, 6.5], tickvals=[1, 2, 3, 4, 5, 6])
    _save(fig, save_dir, f"{dataset}_cb_layer_lr_comp.pdf", html=True)
    return fig


# %%
def plot_cb_layer_lr_comparison(
    layer_lr: pd.DataFrame, colored_region_df: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, *,
    dataset: str = DATASET, save_dir: Path | None = None,
    highlight: list[str] | None = None,
):
    """Per-instance right-layer vs left-layer scatter (legacy nb 9
    `layer_lr_idx_comp` panel)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    fig = _plot_layer_lr_scatter(
        layer_lr, "instance", colored_region_df,
        colored_main_groups_df, highlight=highlight,
    )
    fig.update_xaxes(range=[1.8, 6.5], tickvals=[1, 2, 3, 4, 5, 6, 7])
    fig.update_yaxes(range=[1.8, 6.5], tickvals=[1, 2, 3, 4, 5, 6, 7])
    _save(fig, save_dir, f"{dataset}_cb_layer_lr_idx_comp.pdf", html=True)
    return fig


# %%
def plot_cb_vic_lr_homologue(
    homologue_df: pd.DataFrame, colored_region_df: pd.DataFrame, *,
    vic_thre: float = 5e-4, dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """Per-cell-type right vs left CB VIC scatter + diff histogram
    (legacy nb 9 cells 24-25). Region labels are based on summed VIC:
    `VIC_r+VIC_l < vic_thre` → VNC, `< 1e-2` → left OL, `>= 1e-2` → right OL.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    df = homologue_df.copy()
    df["VIC_diff"] = (df["VIC_r"] - df["VIC_l"]) / (df["VIC_r"] + df["VIC_l"])
    s = df["VIC_r"] + df["VIC_l"]
    sub1 = df[s < vic_thre].copy(); sub1["region"] = "VNC"
    sub2 = df[(s >= vic_thre) & (s < 1e-2)].copy(); sub2["region"] = "left OL"
    sub3 = df[s >= 1e-2].copy(); sub3["region"] = "right OL"
    plot_df = pd.concat([sub1, sub2, sub3], axis=0, ignore_index=True)

    fig = _scatter(plot_df, "VIC_r", "VIC_l", "region", "cell_type",
                   colored_region_df, [])
    fig.add_trace(go.Scatter(
        x=[5e-8, 2], y=[5e-8, 2], mode="lines",
        line=dict(color="gray", dash="dash"), showlegend=False,
    ))
    fig.update_xaxes(type="log", range=[np.log10(5e-8), np.log10(1.2)])
    fig.update_yaxes(type="log", range=[np.log10(5e-8), np.log10(1.2)])
    _save(fig, save_dir, f"{dataset}_cb_vic_lr_comp.pdf", html=True)

    bins = np.linspace(-1.05, 1.05, 22)
    centers = (bins[1:] + bins[:-1]) / 2
    hist_fig = go.Figure()
    for reg, sub in [("VNC", sub1), ("left OL", sub2), ("right OL", sub3)]:
        if sub.empty:
            continue
        counts, _ = np.histogram(sub["VIC_diff"].values, bins=bins)
        hist_fig.add_trace(go.Scatter(
            x=centers, y=counts, mode="lines+markers", name=reg,
            line=dict(color=colored_region_df.loc[reg, "color"], width=2),
        ))
    hist_fig.update_layout(
        xaxis_title="(right - left) / (right + left) VIC",
        yaxis_title="no. cell types",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
        paper_bgcolor="white", plot_bgcolor="white",
        height=400, width=500, font=dict(family="arial", size=14),
    )
    hist_fig.update_xaxes(**_AXIS)
    hist_fig.update_yaxes(**_AXIS)
    _save(hist_fig, save_dir, f"{dataset}_cb_vic_lr_comp_diag.pdf")
    return fig, hist_fig


# %%
def plot_cb_vic_binocularity(
    vic_lr_type: pd.DataFrame, colored_region_df: pd.DataFrame, *,
    vic_thre: float = 5e-4, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Per-instance VIC_r vs VIC_l scatter, colored by binocularity region:
    'VNC' = both below `vic_thre`, 'right OL' = both above, 'left OL' = one
    above one below. Matches legacy nb 9 `vic_lr_binoc` panel.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    df = vic_lr_type.copy()
    df["VIC_diff"] = (df["VIC_r"] - df["VIC_l"]) / (df["VIC_r"] + df["VIC_l"])
    both_below = (df["VIC_r"] < vic_thre) & (df["VIC_l"] < vic_thre)
    both_above = (df["VIC_r"] >= vic_thre) & (df["VIC_l"] >= vic_thre)
    sub1 = df[both_below].copy(); sub1["region"] = "VNC"
    sub2 = df[both_above].copy(); sub2["region"] = "right OL"
    sub3 = df[~both_below & ~both_above].copy(); sub3["region"] = "left OL"
    plot_df = pd.concat([sub1, sub2, sub3], axis=0, ignore_index=True)

    fig = _scatter(plot_df, "VIC_r", "VIC_l", "region", "instance",
                   colored_region_df, [])
    fig.add_trace(go.Scatter(
        x=[5e-8, 2], y=[5e-8, 2], mode="lines",
        line=dict(color="gray", dash="dash"), showlegend=False,
    ))
    fig.update_xaxes(type="log", range=[np.log10(8e-7), np.log10(1)])
    fig.update_yaxes(type="log", range=[np.log10(8e-7), np.log10(1)])
    _save(fig, save_dir, f"{dataset}_cb_vic_lr_binoc.pdf", html=True)

    bins = np.linspace(-1.05, 1.05, 22)
    hist_fig = go.Figure()
    for reg, sub in [("VNC", sub1), ("right OL", sub2), ("left OL", sub3)]:
        if sub.empty:
            continue
        counts, _ = np.histogram(sub["VIC_diff"].values, bins=bins)
        centers = (bins[1:] + bins[:-1]) / 2
        color = colored_region_df.loc[reg, "color"]
        hist_fig.add_trace(go.Scatter(
            x=centers, y=counts, mode="lines+markers", name=reg,
            line=dict(color=color, width=2),
        ))
    hist_fig.update_layout(
        xaxis_title="(right - left) / (right + left) VIC",
        yaxis_title="no. cell types",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
        paper_bgcolor="white", plot_bgcolor="white",
        height=400, width=500, font=dict(family="arial", size=14),
    )
    hist_fig.update_xaxes(**_AXIS)
    hist_fig.update_yaxes(**_AXIS)
    _save(hist_fig, save_dir, f"{dataset}_cb_vic_lr_binoc_diag.pdf")
    return fig, hist_fig


# %% [markdown]
# === 10. CB RFs ===


# %%
def plot_cb_rf_r2_summary(
    type_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, r2_thre: float = 0.05, dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """r2 histogram (vertical), r2 cumulative, and r2-vs-hitting_time scatter
    across regions — legacy nb 10 r2 panel."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    figs = []
    bins = np.linspace(-1.05, 1.05, 31)

    fig = _hist_lines_v(type_df, "r2", "region", bins, colored_region_df,
                        height=400, width=300)
    fig.update_layout(
        yaxis_title="ARF r2", xaxis_title="frac. cell types",
        xaxis_range=[-.01, 0.5], xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_yaxes(range=[-1.1, 1.1])
    figs.append((fig, f"{dataset}_cb_r2_hist.pdf"))

    bins2 = np.linspace(-1.05, 1.05, 22)
    fig = _hist_cumsum_lines_v(type_df, "r2", "region", bins2,
                               colored_region_df, height=400, width=300)
    fig.add_hline(y=r2_thre, line_dash="dash", line_color="gray")
    fig.update_layout(
        yaxis_title="ARF r2", xaxis_title="cum. frac. cell types",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_yaxes(range=[-1.1, 1.1])
    fig.update_xaxes(range=[-0.1, 1.1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_cb_r2_cumsum.pdf"))

    fig = _scatter(type_df, "hitting_time", "r2", "region", "instance",
                   colored_region_df, [], height=600, width=600)
    fig.add_hline(y=r2_thre, line_dash="dash", line_color="gray")
    fig.update_yaxes(range=[-1.1, 1.1])
    fig.update_xaxes(range=[0.8, 5.2])
    fig.update_layout(xaxis_title="layer", yaxis_title="ARF r2",
                      xaxis_title_standoff=0, yaxis_title_standoff=2)
    figs.append((fig, f"{dataset}_cb_layer_vs_r2.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_cb_rf_size_summary(
    type_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, r2_thre: float = 0.05, vic_thre: float = 5e-4,
    star_types: list[str] | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Size histogram + size-vs-layer, VIC-vs-size and VIC-vs-amp scatters
    (legacy nb 10 size panel). Only includes rows with `r2 > r2_thre`."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    stars = [s + "_R" for s in _RF_STAR_EXAMPLES] if star_types is None else list(star_types)
    figs = []
    sub = type_df[type_df["r2"] > r2_thre].copy()

    bins = 2 ** np.linspace(np.log2(1.4), np.log2(350), 12)
    fig = _hist_lines_v(sub, "size", "region", bins, colored_region_df,
                        height=400, width=300)
    fig.update_layout(
        yaxis_title="ARF size", xaxis_title="frac. cell types",
        xaxis_range=[-.01, 0.5], xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_yaxes(type="log", tickformat="d",
                     tickvals=[5, 10, 50, 100], range=[np.log10(1.45), np.log10(350)])
    figs.append((fig, f"{dataset}_cb_size_hist.pdf"))

    fig = _scatter(sub, "hitting_time", "size", "region", "instance",
                   colored_region_df, stars, height=600, width=600)
    fig.update_yaxes(type="log", tickformat="d",
                     tickvals=[5, 10, 50, 100], range=[np.log10(1.45), np.log10(350)])
    fig.update_xaxes(range=[0.8, 5.2])
    fig.update_layout(xaxis_title="layer", yaxis_title="ARF size",
                      xaxis_title_standoff=0, yaxis_title_standoff=2)
    figs.append((fig, f"{dataset}_cb_layer_vs_size.pdf"))

    fig = _scatter(sub, "VIC", "size", "region", "instance",
                   colored_region_df, stars, height=600, width=600)
    fig.add_vline(x=vic_thre, line_dash="dash", line_color="gray")
    fig.update_yaxes(type="log", tickformat="d",
                     tickvals=[5, 10, 50, 100], range=[np.log10(3), np.log10(350)])
    fig.update_xaxes(type="log", tickvals=[0.001, 0.01, 0.1],
                     range=[np.log10(4e-4), np.log10(0.6)])
    fig.update_layout(yaxis_title="ARF size", xaxis_title="VIC",
                      xaxis_title_standoff=0, yaxis_title_standoff=2)
    figs.append((fig, f"{dataset}_cb_vic_vs_size.pdf"))

    fig = _scatter(sub, "VIC", "amp", "region", "instance",
                   colored_region_df, stars, height=600, width=600)
    fig.add_vline(x=vic_thre, line_dash="dash", line_color="gray")
    fig.update_yaxes(type="log", range=[np.log10(1e-6), np.log10(1e-1)])
    fig.update_xaxes(type="log", tickvals=[0.001, 0.01, 0.1],
                     range=[np.log10(4e-4), np.log10(0.6)])
    fig.update_layout(yaxis_title="ARF amp", xaxis_title="VIC",
                      xaxis_title_standoff=0, yaxis_title_standoff=2)
    figs.append((fig, f"{dataset}_cb_vic_vs_amp.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_rf_size_per_layer(
    type_df: pd.DataFrame,
    *, color: str = "#333333", col_to_deg_size: float = 25.0,
    dataset: str = DATASET, save_dir: Path | None = None,
    width: int = 896, height: int = 280, font_size: float = 15.12,
):
    """Single-box-per-layer box plot of ARF size vs hitting-time layer.
    Size in deg² (col² × `col_to_deg_size`, default 25 — same factor used by
    `plot_rf_size_vs_experiment`). Bin width 0.5, centers at multiples of 0.5
    (0, 0.5, ..., 6). Pass OL + VPN + VCBN combined. Legacy nb 10."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    bins = np.arange(-0.25, 6 + 0.25 + 1e-9, 0.5)
    labels = (bins[1:] + bins[:-1]) / 2
    df = type_df.copy()
    df["bin"] = pd.cut(df["hitting_time"], bins=bins, labels=labels).astype(float)
    df = df.dropna(subset=["bin", "size"])
    df["size_deg"] = df["size"] * col_to_deg_size

    fig = go.Figure()
    fig.add_trace(go.Box(
        x=df["bin"], y=df["size_deg"],
        marker=dict(color=color), line=dict(color=color),
    ))
    fig.update_layout(
        xaxis_title="layer", yaxis_title="ARF size (deg²)",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="arial", size=font_size),
        height=height, width=width, showlegend=False,
        xaxis_range=[0.5, 5.5], boxmode="group",
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(type="log", tickformat="d",
                     tickvals=[50, 100, 500, 1000, 5000],
                     range=[np.log10(25), np.log10(8750)],
                     title_standoff=2, **_AXIS)
    _save(fig, save_dir, f"{dataset}_size_per_layer.pdf", html=True)
    return fig


# %%
def plot_ol_to_cb_heatmap(
    tb_clu_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    height: int = 500, width: int = 1050, title_suffix: str = "",
    xlabel: str = "CB types (small <---> medium)",
    ylabel: str = "OL output types (medium <---> small)",
    show_ticks: bool = False,
):
    """Heatmap of normalised OL-output → CB-type effective weights. Legacy
    nb 10 cell 1d2d8ed4 / e14975dd / 722c0f59."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    fig = go.Figure(data=go.Heatmap(
        z=tb_clu_df.values,
        x=tb_clu_df.columns.tolist(), y=tb_clu_df.index.tolist(),
        colorscale="Greys", reversescale=False, showscale=True,
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        font=dict(family="arial", size=18),
        title=f"rel. effective weight from OL output to CB types{title_suffix}",
        xaxis_title=xlabel, yaxis_title=ylabel,
        xaxis_tickangle=-45 if show_ticks else 0,
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width,
    )
    if show_ticks:
        fig.update_xaxes(
            showline=True, title_standoff=0, linewidth=1, linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            showline=True, title_standoff=2, linewidth=1, linecolor="black",
            mirror=True, autorange="reversed",
        )
    else:
        fig.update_xaxes(
            showline=True, title_standoff=2, linewidth=1, linecolor="black",
            mirror=True, tickvals=list(range(tb_clu_df.shape[1])),
            ticktext=["" for _ in tb_clu_df.columns],
        )
        fig.update_yaxes(
            showline=True, title_standoff=2, linewidth=1, linecolor="black",
            mirror=True, tickvals=list(range(tb_clu_df.shape[0])),
            ticktext=["" for _ in tb_clu_df.index],
            autorange="reversed",
        )
    _save(fig, save_dir, f"{dataset}_ol_to_cb_heatmap.pdf")
    return fig


# %%
def plot_ol_roi_coverage_per_roi(
    coverage: dict, sectors: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None, cov0_max: float = 0.4,
):
    """For each ROI in `coverage['per_roi']`, save three PDFs: full hex heatmap,
    per-sector relative heatmap, and eye-symbol pictogram. Iterates ~78 ROIs.
    `cov0_max`: upper bound for the relative-coverage colorscale (values above
    are clipped to full colour)."""
    from connectome_interpreter.external_map import hex_heatmap
    save_dir = save_dir if save_dir is not None else FIG_DIR / "coverage"
    summary = coverage["summary"].set_index("roi")
    valid_sector_coords = sectors.loc[sectors["coords"].notna() & (sectors["coords"] != "None"), "coords"].values
    sector_coords = {
        i + 1: sectors.loc[(sectors.sector == i + 1) & sectors["coords"].notna() & (sectors["coords"] != "None"), "coords"].values
        for i in range(5)
    }

    figs = []
    for roi, roi_df in coverage["per_roi"].items():
        w = roi_df.set_index("coords")[["effective weight"]]
        w = w[w.index.notna() & (w.index.astype(str) != "None")]
        if w.empty:
            continue

        cmax = float(np.quantile(np.abs(w["effective weight"].values), 0.995))
        fig = hex_heatmap(
            w, custom_colorscale="RdBu_r",
            global_min=-cmax, global_max=cmax, dataset="mcns_right",
        )
        figs.append((fig, f"{dataset}_{roi}_coverage_full.pdf"))

        rel = w.reindex(valid_sector_coords).fillna(0).copy()
        for sec_id, coords in sector_coords.items():
            rel.loc[coords, "effective weight"] = rel.loc[coords, "effective weight"].sum()
        fig = hex_heatmap(
            rel, custom_colorscale="RdBu_r",
            global_min=0, global_max=cov0_max, dataset="mcns_right",
        )
        figs.append((fig, f"{dataset}_{roi}_coverage_relative.pdf"))

        sec_vals = summary.loc[roi, ["sec1", "sec2", "sec3", "sec4", "sec5"]].values.astype(float)
        colors = [
            px.colors.sample_colorscale("RdBu_r", float(min(v, cov0_max)) / cov0_max)[0]
            for v in sec_vals
        ]
        fig = _eyesymbol(colors)
        figs.append((fig, f"{dataset}_{roi}_coverage_symbol.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_ol_roi_coverage_summary(
    coverage: dict, *, dataset: str = DATASET, save_dir: Path | None = None,
    top_n_coverage: int = 60, top_n_pval: int = 30, cov0_max: float = 0.4,
):
    """Cross-ROI summary: ranked p-value line plot, relative-coverage heatmap
    (sectors × top-N ROIs, ordered D/V/C/A/P), and p-value heatmap (single row).
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "coverage"
    summary = coverage["summary"].copy()
    figs = []

    ranked = summary.sort_values("pval").reset_index(drop=True)
    ranked["index"] = np.arange(len(ranked))
    fig = _point_line(ranked, "index", "pval", height=300, width=1200)
    fig.update_xaxes(tickvals=ranked["index"], ticktext=ranked["roi"], tickangle=-45)
    fig.update_layout(xaxis_title="ROI", yaxis_title="p-value")
    fig.update_yaxes(type="log", range=[-4.2, -2])
    figs.append((fig, f"{dataset}_pval_ranked.pdf"))

    top = summary.iloc[:top_n_coverage]
    cov = top[["sec2", "sec4", "sec5", "sec1", "sec3"]].T
    cov.index = ["D", "V", "C", "A", "P"]
    cov.columns = top["roi"].values
    fig = go.Figure(data=go.Heatmap(
        z=cov.values, x=cov.columns.tolist(), y=cov.index.tolist(),
        colorscale="RdBu_r", showscale=True, zmin=0, zmax=cov0_max,
    ))
    fig.add_hline(y=1.5, line_color="black", line_width=0.5)
    fig.add_hline(y=2.5, line_color="black", line_width=0.5)
    fig.update_layout(
        font=dict(family="arial", size=16),
        title="relative coverage", xaxis_title="", yaxis_title="",
        paper_bgcolor="white", plot_bgcolor="white",
        height=300, width=1400,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.update_xaxes(tickangle=-45, title_standoff=0, **_AXIS)
    fig.update_yaxes(autorange="reversed", title_standoff=2, **_AXIS)
    figs.append((fig, f"{dataset}_rel_coverage_per_roi.pdf"))

    top_p = summary.iloc[:top_n_pval]
    pv = np.log10(top_p["pval"].values.astype(float)).reshape(1, -1)
    fig = go.Figure(data=go.Heatmap(
        z=pv, x=top_p["roi"].values.tolist(), y=["log10 p"],
        colorscale="Viridis_r", showscale=True, zmin=-4, zmax=0,
    ))
    fig.update_layout(
        font=dict(family="arial", size=16),
        title="p-values maxT", xaxis_title="", yaxis_title="",
        paper_bgcolor="white", plot_bgcolor="white",
        height=200, width=800,
    )
    fig.update_xaxes(tickangle=-45, title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    figs.append((fig, f"{dataset}_pval_per_roi.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %% [markdown]
# === 11. Model comparisons ===
