"""Map files produced by figures.ipynb plots onto per-panel filenames
defined by paper_figs.xlsx, and distribute them into results/<folder>/.

Naming rules (per panel, after dropping Blender/Neuroglancer/custom/AZ rows):
  - 1 file in panel, no position label  -> <folder>_<panel>.<ext>
  - >1 files, no position label         -> <folder>_<panel>_<i>.<ext>
  - 1 file under a position             -> <folder>_<panel>_<position>.<ext>
  - >1 files under a position           -> <folder>_<panel>_<position>_<i>.<ext>

`folder` is "main_<N>" or "si_<N>"; `position` is left/right/top/bottom.
A figure label may omit the panel letter (e.g. "SI Fig. 10"); then `<panel>`
is dropped from the name (-> <folder>.<ext> or <folder>_<i>.<ext>).
Numbering follows xlsx row order. Source extension is preserved.

Source rows containing the literal `XXX` are treated as templates: at
distribute time, the topic dir is globbed for the pattern with `XXX`
replaced by `*`, basenames already named elsewhere in the manifest are
excluded, and the remaining files become numbered panels.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

from utils.config import FIG_DATASET, PARAMS_DIR

ROOT = Path(__file__).resolve().parent.parent.parent
XLSX = PARAMS_DIR / "paper_figs.xlsx"
RESULTS = ROOT / "results"
HTML_DIR = RESULTS / "html_figures"

EXTERNAL = {"Blender", "Neuroglancer", "custom", "AZ"}

_LABEL_RE = re.compile(
    r"^(Main|SI)\s+Fig\.\s+(\d+)([a-z])?(?:\s+(left|right|top|bottom))?$"
)

_DATASET_TOKEN_RE = re.compile(r"malecns_v\d+\.\d+")


def _parse_label(label: str) -> tuple[str, str, str | None]:
    m = _LABEL_RE.match(str(label).strip())
    if not m:
        raise ValueError(f"unrecognised paper-figure label: {label!r}")
    kind, num, panel, position = m.groups()
    return f"{'main' if kind == 'Main' else 'si'}_{num}", panel, position


def _stem(folder: str, panel: str | None, position: str | None, n: int, i: int) -> str:
    base = "_".join(p for p in (folder, panel, position) if not pd.isna(p))
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
    for (folder, panel), grp in fixed.groupby(["folder", "panel"], sort=False, dropna=False):
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


def _place(src: Path, dst: Path, *, mode: str) -> None:
    """Copy or move a single file, creating the destination directory."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    (shutil.copy2 if mode == "copy" else shutil.move)(str(src), str(dst))


def _distribute_sidecar(candidates: list[Path], dst: Path, *, mode: str) -> bool:
    """Place the first existing ``.html`` sidecar among ``candidates`` into
    ``HTML_DIR`` under the panel target name. Returns True if one was placed.

    Sidecar distribution is intentionally independent of the main panel file: a
    sidecar is (re)populated into ``html_figures/`` whenever it exists on disk —
    even when the main file is already in place (so a deleted ``html_figures/``
    can be rebuilt by re-running :func:`distribute`) or when a sidecar exists
    without its main file (e.g. the PDF was not regenerated)."""
    for src_html in candidates:
        if src_html.exists():
            HTML_DIR.mkdir(parents=True, exist_ok=True)
            _place(src_html, HTML_DIR / dst.with_suffix(".html").name, mode=mode)
            return True
    return False


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
    drain topic folders. Plotly `.html` sidecars are distributed into
    `html_figures/` independently of the main file (so a deleted
    `html_figures/` is rebuilt even when the panel PDFs are already present).

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
            _place(topic_src, dst, mode=mode)
            status = mode + "d-from-topic"
        elif panel_src.exists() and panel_src != dst:
            _place(panel_src, dst, mode="move")
            status = "renamed-in-place"
        elif panel_src == dst and dst.exists():
            status = "already-present"
        # Sidecar handling is independent of the main file (see
        # _distribute_sidecar): this repopulates html_figures/ from existing
        # sidecars even when the panel PDF is already present or absent.
        html_placed = _distribute_sidecar(
            [topic_src.with_suffix(".html"), panel_src.with_suffix(".html")],
            dst, mode=mode,
        )
        rows.append({
            "source": r["source"],
            "target": f"{r['folder']}/{r['target_name']}",
            "status": status,
            "html": html_placed,
        })

    out = pd.DataFrame(rows)
    if verbose:
        scope = folder or "ALL"
        counts = out["status"].value_counts().to_dict()
        n_html = int(out["html"].sum())
        print(f"[distribute {scope}] {counts} | html_sidecars={n_html}")
        miss = out[out["status"] == "missing"]
        if len(miss):
            head = "\n  ".join(miss["source"].head(10).tolist())
            tail = f"\n  ... (+{len(miss) - 10} more)" if len(miss) > 10 else ""
            print(f"  missing sources:\n  {head}{tail}")


# Viewer template for build_index(). The figures list is injected at
# `__FIGURES_JSON__`; everything else (CSS / JS) matches the hand-tuned
# committed viewer. Panels are grouped by their `main_<N>` / `si_<N>` prefix.
_INDEX_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interactive HTML Figures</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #18202a;
      --muted: #647082;
      --line: #d9dee7;
      --accent: #0f6b57;
      --accent-soft: #e4f2ee;
      --shadow: 0 1px 2px rgb(20 32 48 / 8%);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }

    main {
      width: min(1400px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 32px;
    }

    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 24px;
    }

    h1 {
      margin: 0;
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: 0;
    }

    .count {
      color: var(--muted);
      font-size: 0.95rem;
      white-space: nowrap;
    }

    .groups {
      display: grid;
      gap: 16px;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }

    .sidebar {
      display: grid;
      gap: 16px;
      max-height: calc(100vh - 128px);
      overflow: auto;
      padding-right: 2px;
    }

    section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    h2 {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin: 0;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 1rem;
      letter-spacing: 0;
    }

    h2 span {
      color: var(--muted);
      font-size: 0.85rem;
      font-weight: 500;
    }

    ul {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1px;
      margin: 0;
      padding: 0;
      list-style: none;
      background: var(--line);
    }

    li {
      min-width: 0;
      background: var(--panel);
    }

    a {
      display: block;
      min-height: 48px;
      padding: 13px 16px;
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 0.92rem;
      text-decoration: none;
      overflow-wrap: anywhere;
    }

    a:hover,
    a:focus {
      background: var(--accent-soft);
      color: var(--accent);
      outline: none;
    }

    a.active {
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
    }

    .viewer {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      min-height: calc(100vh - 128px);
      overflow: hidden;
    }

    .viewer-header {
      min-height: 48px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
    }

    .viewer-title {
      min-width: 0;
      margin: 0;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 3.68rem;
      font-weight: 600;
      overflow-wrap: anywhere;
    }

    iframe {
      display: block;
      width: 50%;
      height: calc(50vh - 88.5px);
      min-height: 310px;
      border: 0;
      transform: scale(2);
      transform-origin: top left;
    }

    @media (max-width: 640px) {
      main {
        width: min(100% - 24px, 980px);
        padding-top: 24px;
      }

      header {
        align-items: stretch;
        flex-direction: column;
        gap: 10px;
      }

      h1 {
        font-size: 1.55rem;
      }

      .count {
        white-space: normal;
      }

      .layout {
        grid-template-columns: 1fr;
      }

      .sidebar {
        max-height: none;
        overflow: visible;
        padding-right: 0;
      }

      .viewer,
      iframe {
        min-height: 260px;
      }

      iframe {
        height: 35vh;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Interactive HTML Figures</h1>
      <div class="count" id="panel-count"></div>
    </header>

    <div class="layout">
      <div class="sidebar">
        <div class="groups" id="figure-groups"></div>
      </div>

      <section class="viewer" aria-label="Selected panel">
        <div class="viewer-header">
          <p class="viewer-title" id="viewer-title"></p>
        </div>
        <iframe id="viewer-frame" title="Selected interactive figure"></iframe>
      </section>
    </div>
  </main>

  <script>
    const figures = __FIGURES_JSON__;

    const groups = figures.reduce((acc, file) => {
      const group = file.match(/^(?:main|si)_\d+/)?.[0] || "other";
      (acc[group] ||= []).push(file);
      return acc;
    }, {});

    const groupRoot = document.querySelector("#figure-groups");
    const count = document.querySelector("#panel-count");
    const frame = document.querySelector("#viewer-frame");
    const title = document.querySelector("#viewer-title");
    const linksByFile = new Map();

    function normalizeFrame() {
      let attempts = 0;

      function tryNormalize() {
        attempts += 1;

        try {
          const win = frame.contentWindow;
          const doc = frame.contentDocument || win?.document;

          if (!doc) {
            return;
          }

          let style = doc.querySelector("#viewer-normalize-style");
          if (!style) {
            style = doc.createElement("style");
            style.id = "viewer-normalize-style";
            style.textContent = `
              html,
              body {
                width: 100% !important;
                height: 100% !important;
                margin: 0 !important;
                overflow: hidden !important;
              }

              body {
                display: grid !important;
                place-items: center !important;
              }

              body > div:first-child {
                width: 100% !important;
                height: 100% !important;
              }

              .plotly-graph-div,
              .svg-container {
                width: 100% !important;
                height: 100% !important;
              }
            `;
            (doc.head || doc.documentElement).append(style);
          }

          const graphs = Array.from(doc.querySelectorAll(".plotly-graph-div"));
          graphs.forEach((graph) => {
            graph.style.width = "100%";
            graph.style.height = "100%";
          });

          if (win?.Plotly && graphs.length > 0) {
            let resized = false;
            graphs.forEach((graph) => {
              if (!graph._fullLayout) {
                return;
              }

              win.Plotly.relayout(graph, { autosize: true });
              win.Plotly.Plots?.resize?.(graph);
              resized = true;
            });

            if (resized) {
              return;
            }
          }
        } catch {
          return;
        }

        if (attempts < 20) {
          setTimeout(tryNormalize, 100);
        }
      }

      tryNormalize();
    }

    function selectPanel(file, updateHash = true) {
      frame.src = file;
      title.textContent = file.replace(/\.html$/, "");

      linksByFile.forEach((link, linkFile) => {
        link.classList.toggle("active", linkFile === file);
        if (linkFile === file) {
          link.setAttribute("aria-current", "true");
        } else {
          link.removeAttribute("aria-current");
        }
      });

      if (updateHash) {
        history.replaceState(null, "", `#${file.replace(/\.html$/, "")}`);
      }
    }

    function render() {
      groupRoot.replaceChildren();
      linksByFile.clear();

      let panelCount = 0;

      Object.entries(groups).forEach(([group, files]) => {
        panelCount += files.length;

        const section = document.createElement("section");
        const heading = document.createElement("h2");
        const label = document.createElement("span");
        const list = document.createElement("ul");

        heading.append(group.replace("_", " "));
        label.textContent = `${files.length} panel${files.length === 1 ? "" : "s"}`;
        heading.append(label);

        files.forEach((file) => {
          const item = document.createElement("li");
          const link = document.createElement("a");
          link.href = file;
          link.textContent = file.replace(/\.html$/, "");
          link.addEventListener("click", (event) => {
            event.preventDefault();
            selectPanel(file);
          });
          item.append(link);
          list.append(item);
          linksByFile.set(file, link);
        });

        section.append(heading, list);
        groupRoot.append(section);
      });

      count.textContent = `${panelCount} panel${panelCount === 1 ? "" : "s"}`;
    }

    frame.addEventListener("load", normalizeFrame);
    window.addEventListener("resize", normalizeFrame);

    render();
    const hashPanel = decodeURIComponent(location.hash.replace(/^#/, ""));
    const initialPanel = hashPanel.endsWith(".html") ? hashPanel : `${hashPanel}.html`;
    selectPanel(figures.includes(initialPanel) ? initialPanel : figures[0], false);
  </script>
</body>
</html>
"""


_PANEL_NAME_RE = re.compile(r"^(?:main|si)_\d+")


def _index_sort_key(name: str) -> tuple:
    """Natural sort: main before si, then figure number, then panel tokens
    (digits compared numerically) so e.g. main_4_d_2 follows main_4_d_1 and
    si_* groups follow main_* groups."""
    stem = name[:-5] if name.endswith(".html") else name
    m = re.match(r"^(main|si)_(\d+)(.*)$", stem)
    if not m:
        return (2, 0, [(1, stem)])
    kind, num, rest = m.group(1), int(m.group(2)), m.group(3)
    tokens = [
        (0, int(tok)) if tok.isdigit() else (1, tok)
        for tok in rest.split("_") if tok
    ]
    return (0 if kind == "main" else 1, num, tokens)


def build_index(*, html_dir: Path = HTML_DIR) -> Path:
    """(Re)generate `html_figures/index.html` listing every distributed `.html`
    panel currently in `html_dir`. Run after :func:`distribute` so the viewer
    matches the distributed figure set. Only panel-named files
    (`main_<N>...` / `si_<N>...`) are listed, so stray sidecars do not pollute
    the viewer. Returns the written index path."""
    files = sorted(
        (
            p.name for p in html_dir.glob("*.html")
            if p.name != "index.html" and _PANEL_NAME_RE.match(p.name)
        ),
        key=_index_sort_key,
    )
    html = _INDEX_TEMPLATE.replace("__FIGURES_JSON__", json.dumps(files, indent=6))
    out = html_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"[build_index] wrote {out.name} listing {len(files)} panels")
    return out


def parity_check(
    *, results_dir: Path = RESULTS, exclude_folders: tuple[str, ...] | list[str] = (),
) -> None:
    """Errors if any expected panel file is missing. For XXX templates, errors
    if zero files matched the pattern. `exclude_folders` skips whole figure
    folders (e.g. panels produced by other pipelines and not distributed here)."""
    exclude = set(exclude_folders)
    static = _build_manifest()
    expanded = _expand_templates(results_dir=results_dir)
    templates = _templates()
    if exclude:
        static = static[~static["folder"].isin(exclude)].reset_index(drop=True)
        expanded = expanded[~expanded["folder"].isin(exclude)].reset_index(drop=True)
        templates = templates[~templates["folder"].isin(exclude)].reset_index(drop=True)

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
