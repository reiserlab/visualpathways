# Optic Lobe Connectome

This repository is a collection of code for analyzing the optic lobe in the new Male Brain dataset ([Janelia FlyEM](https://neuprint-cns.janelia.org/?dataset=cns&qt=findneurons)). At this point it is only intended for internal use -- if you can see the repository, the [Reiser lab](https://www.janelia.org/lab/reiser-lab/) has invited you to contribute to the effort.

Some code was carried over from the [male drosophila visual system connectome code](https://github.com/reiserlab/male-drosophila-visual-system-connectome-code).

## Interactive figures

Interactive HTML figures are available through the [GitHub Pages figure browser](https://reiserlab.github.io/ol-connectome-analysis/results/html_figures/).

## Preprint pipeline

The preprint analysis pipeline (orchestrator notebooks that reproduce every
paper/SI panel) lives under [`src/judith/`](src/judith/). See
[`src/judith/README.md`](src/judith/README.md) for its layout, setup, and
run order.

## Install

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

Dependencies are declared in [`pyproject.toml`](pyproject.toml);
`requirements.txt` just re-exports that list for editable install plus
dev tools. Copy `.env_sample` to `.env` and fill in neuprint credentials
before running any notebook that hits the server.