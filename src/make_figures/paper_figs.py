"""Map files produced by figures.ipynb plots onto per-panel filenames
defined by paper_figs.xlsx, and distribute them into results/<folder>/.

Naming rules (per panel, after dropping Blender/Neuroglancer/custom rows):
  - 1 file in panel, no position label  -> <folder>_<panel>.<ext>
  - >1 files, no position label         -> <folder>_<panel>_<i>.<ext>
  - 1 file under a position             -> <folder>_<panel>_<position>.<ext>
  - >1 files under a position           -> <folder>_<panel>_<position>_<i>.<ext>

`folder` is "main_<N>" or "si_<N>"; `position` is left/right/top/bottom.
Numbering follows xlsx row order. Source extension is preserved.

Source rows containing the literal `XXX` are treated as templates: at
distribute time, the topic dir is globbed for the pattern with `XXX`
replaced by `*`, basenames already named elsewhere in the manifest are
excluded, and the remaining files become numbered panels.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

from utils.config import FIG_DATASET, PARAMS_DIR

ROOT = Path(__file__).resolve().parent.parent.parent
XLSX = PARAMS_DIR / "paper_figs.xlsx"
RESULTS = ROOT / "results"
HTML_DIR = RESULTS / "html_figures"

EXTERNAL = {"Blender", "Neuroglancer", "custom"}

_LABEL_RE = re.compile(
    r"^(Main|SI)\s+Fig\.\s+(\d+)([a-z])(?:\s+(left|right|top|bottom))?$"
)

_DATASET_TOKEN_RE = re.compile(r"malecns_v\d+\.\d+")


def _parse_label(label: str) -> tuple[str, str, str | None]:
    m = _LABEL_RE.match(str(label).strip())
    if not m:
        raise ValueError(f"unrecognised paper-figure label: {label!r}")
    kind, num, panel, position = m.groups()
    return f"{'main' if kind == 'Main' else 'si'}_{num}", panel, position


def _stem(folder: str, panel: str, position: str | None, n: int, i: int) -> str:
    base = f"{folder}_{panel}" if position is None else f"{folder}_{panel}_{position}"
    return base if n == 1 else f"{base}_{i}"


def _parse_xlsx() -> pd.DataFrame:
    df = pd.read_excel(XLSX)
    parsed = [_parse_label(lab) for lab in df["paper figure"]]
    df = df.assign(
        folder=[p[0] for p in parsed],
        panel=[p[1] for p in parsed],
        position=[p[2] for p in parsed],
    )
    df["source"] = df["results file"].astype(str)
    pipe = df[~df["source"].isin(EXTERNAL)].copy().reset_index(drop=True)
    # Rebind any hard-coded dataset token to the configured figure filename
    # prefix so the manifest survives data-source dataset bumps.
    pipe["source"] = pipe["source"].str.replace(
        _DATASET_TOKEN_RE, FIG_DATASET, regex=True,
    )
    pipe["source_basename"] = [Path(p).name for p in pipe["source"]]
    pipe["source_ext"] = [Path(p).suffix for p in pipe["source"]]
    pipe["position"] = pipe["position"].astype(object).where(
        pipe["position"].notna(), None
    )
    pipe["is_template"] = pipe["source_basename"].str.contains("XXX")
    return pipe


def folder_order() -> list[str]:
    df = pd.read_excel(XLSX)
    return list(dict.fromkeys(_parse_label(lab)[0] for lab in df["paper figure"]))


def _build_manifest() -> pd.DataFrame:
    """Static manifest: one row per non-external, non-template xlsx row."""
    pipe = _parse_xlsx()
    fixed = pipe[~pipe["is_template"]].copy()

    target_by_idx: dict[int, str] = {}
    for (folder, panel), grp in fixed.groupby(["folder", "panel"], sort=False):
        positions = grp["position"].tolist()
        has_position = any(p is not None for p in positions)
        all_positioned = all(p is not None for p in positions)
        if has_position and not all_positioned:
            raise ValueError(
                f"panel {folder}_{panel} mixes positioned and non-positioned rows"
            )
        if not has_position:
            n = len(grp)
            for i, (idx, row) in enumerate(grp.iterrows(), 1):
                target_by_idx[idx] = f"{_stem(folder, panel, None, n, i)}{row['source_ext']}"
        else:
            for pos in dict.fromkeys(positions):
                sub = grp[grp["position"] == pos]
                n = len(sub)
                for i, (idx, row) in enumerate(sub.iterrows(), 1):
                    target_by_idx[idx] = f"{_stem(folder, panel, pos, n, i)}{row['source_ext']}"
    fixed["target_name"] = [target_by_idx[i] for i in fixed.index]
    return fixed[
        ["source", "source_basename", "source_ext",
         "folder", "panel", "position", "target_name"]
    ].reset_index(drop=True)


def _templates() -> pd.DataFrame:
    """Template rows (those with XXX in source). One row per template panel."""
    pipe = _parse_xlsx()
    return pipe[pipe["is_template"]][
        ["source", "source_basename", "source_ext", "folder", "panel", "position"]
    ].reset_index(drop=True)


def _expand_templates(*, results_dir: Path = RESULTS) -> pd.DataFrame:
    """Resolve each XXX template by globbing the topic dir at distribute time.
    Excludes basenames already in the static manifest (so files used in other
    positions of the same / different panels are not double-counted).

    The text that `XXX` matched in each file is extracted and used as the
    panel suffix — e.g. `<folder>_<panel>_<position>_<ROI>.pdf` — instead of
    sequential indices. This makes the panel name self-documenting.
    """
    static = _build_manifest()
    templates = _templates()
    used = set(static["source_basename"])
    rows = []
    for _, t in templates.iterrows():
        src = Path(t["source"])
        topic_dir = results_dir / src.parent
        pattern = src.name.replace("XXX", "*")
        rgx = re.compile("^" + re.escape(src.name).replace(re.escape("XXX"), "(.+)") + "$")
        matches: list[tuple[str, str]] = []
        if topic_dir.exists():
            for p in sorted(topic_dir.glob(pattern)):
                if p.name in used:
                    continue
                m = rgx.match(p.name)
                if m:
                    matches.append((p.name, m.group(1)))
        for basename, roi in matches:
            base = (
                f"{t['folder']}_{t['panel']}"
                if t["position"] is None
                else f"{t['folder']}_{t['panel']}_{t['position']}"
            )
            rows.append({
                "source": str(src.parent / basename),
                "source_basename": basename,
                "source_ext": t["source_ext"],
                "folder": t["folder"],
                "panel": t["panel"],
                "position": t["position"],
                "target_name": f"{base}_{roi}{t['source_ext']}",
            })
    return pd.DataFrame(rows, columns=static.columns)


def manifest(*, expand: bool = True, results_dir: Path = RESULTS) -> pd.DataFrame:
    """Full manifest. If `expand`, XXX template rows are resolved by globbing
    the topic dir on disk; otherwise only static rows are returned."""
    static = _build_manifest()
    if not expand:
        return static
    expanded = _expand_templates(results_dir=results_dir)
    return pd.concat([static, expanded], ignore_index=True)


def _move_with_html(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    op = shutil.copy2 if mode == "copy" else shutil.move
    op(str(src), str(dst))
    src_html = src.with_suffix(".html")
    if src_html.exists():
        HTML_DIR.mkdir(parents=True, exist_ok=True)
        op(str(src_html), str(HTML_DIR / dst.with_suffix(".html").name))


def distribute(
    folder: str | None = None,
    *,
    results_dir: Path = RESULTS,
    mode: str = "copy",
    verbose: bool = True,
) -> None:
    """Copy (or move) source files into per-panel files under
    `results/<folder>/<target_name>`.

    For each manifest row, the source file is searched in this order:
      1. `results/<source>`              (default save_dir = topic folder)
      2. `results/<folder>/<basename>`   (save_dir was redirected)
    Whichever exists is used.

    If `mode='copy'` (default), the source remains in place. Use 'move' to
    drain topic folders. .html sidecars (Plotly) are handled alongside.

    Verbose summary is printed; nothing is returned (keeps notebook cells
    free of an auto-displayed status DataFrame).
    """
    if mode not in ("copy", "move"):
        raise ValueError(f"mode must be 'copy' or 'move', got {mode!r}")
    mf = manifest(expand=True, results_dir=results_dir)
    if folder is not None:
        mf = mf[mf["folder"] == folder].copy()

    rows = []
    for _, r in mf.iterrows():
        dst = results_dir / r["folder"] / r["target_name"]
        topic_src = results_dir / r["source"]
        panel_src = results_dir / r["folder"] / r["source_basename"]
        status = "missing"
        if dst.exists() and not topic_src.exists() and not panel_src.exists():
            status = "already-present"
        elif topic_src.exists():
            _move_with_html(topic_src, dst, mode=mode)
            status = mode + "d-from-topic"
        elif panel_src.exists() and panel_src != dst:
            _move_with_html(panel_src, dst, mode="move")
            status = "renamed-in-place"
        elif panel_src == dst and dst.exists():
            status = "already-present"
        rows.append({"source": r["source"], "target": f"{r['folder']}/{r['target_name']}", "status": status})

    out = pd.DataFrame(rows)
    if verbose:
        scope = folder or "ALL"
        counts = out["status"].value_counts().to_dict()
        print(f"[distribute {scope}] {counts}")
        miss = out[out["status"] == "missing"]
        if len(miss):
            head = "\n  ".join(miss["source"].head(10).tolist())
            tail = f"\n  ... (+{len(miss) - 10} more)" if len(miss) > 10 else ""
            print(f"  missing sources:\n  {head}{tail}")


def parity_check(*, results_dir: Path = RESULTS) -> None:
    """Errors if any expected panel file is missing. For XXX templates, errors
    if zero files matched the pattern."""
    static = _build_manifest()
    expanded = _expand_templates(results_dir=results_dir)
    templates = _templates()

    missing = [
        f"{r['folder']}/{r['target_name']}"
        for _, r in pd.concat([static, expanded], ignore_index=True).iterrows()
        if not (results_dir / r["folder"] / r["target_name"]).exists()
    ]

    template_misses = []
    for _, t in templates.iterrows():
        n = (expanded["folder"] == t["folder"]).sum() and (
            (expanded["folder"] == t["folder"])
            & (expanded["panel"] == t["panel"])
            & (expanded["position"] == t["position"])
        ).sum()
        if not n:
            template_misses.append(
                f"{t['folder']}_{t['panel']}_{t['position']} "
                f"(template {t['source_basename']})"
            )

    if missing or template_misses:
        msg = []
        if missing:
            head = "\n  ".join(missing[:20])
            more = f"\n  ... (+{len(missing) - 20} more)" if len(missing) > 20 else ""
            msg.append(f"{len(missing)} missing target file(s):\n  {head}{more}")
        if template_misses:
            msg.append(
                f"{len(template_misses)} template panel(s) had no matching files:\n  "
                + "\n  ".join(template_misses)
            )
        raise FileNotFoundError("parity check failed:\n" + "\n".join(msg))

    n_static = len(static)
    n_expanded = len(expanded)
    print(
        f"parity check OK: {n_static} static panels + "
        f"{n_expanded} template-expanded panels present"
    )
