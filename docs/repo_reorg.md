# Repository Reorganization Roadmap

This document tracks structural improvements to the `ol-connectome-analysis` repository. Each item includes the problem, the fix, and an effort estimate.

---

## Priority 1: Convert to a Proper Python Package

**Effort:** Medium

**Problem:** Every script (~30 files) copy-pastes the same boilerplate to make imports work:

```python
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
```

There are no `__init__.py` files anywhere under `src/`, so the code is not installable or importable as a package.

**Fix:**
- Add a `pyproject.toml` at the repo root
- Add `__init__.py` files to `src/utils/`, `src/queries/`, `src/quan_propagation/`, `src/network/`, etc.
- Make the project installable with `pip install -e .`
- Remove all `sys.path.append` boilerplate from every file
- Update imports from `from utils.x import y` to `from ol_connectome.utils.x import y` (or chosen package name)

---

## Priority 2: Pin Git Dependencies to Commit SHAs

**Effort:** Small

**Problem:** `requirements.txt` tracks two forked packages at `@master`:

```
navis @ git+https://github.com/artxz/navis.git@master
neuprint_python @ git+https://github.com/artxz/neuprint-python.git@master
```

Any upstream change can silently break the build. Builds are not reproducible.

**Fix:**
- Pin both to a specific commit SHA (e.g. `@a1b2c3d`)
- Add a comment explaining why forks are used instead of upstream packages
- Consider periodically rebasing forks and updating the pinned SHA

---

## Priority 3: Add Linting and CI

**Effort:** Small

**Problem:** No linter configuration, no CI pipeline, no automated checks on pull requests.

**Fix:**
- Add a `ruff` configuration (in `pyproject.toml` or `ruff.toml`)
- Create a `.github/workflows/ci.yml` that runs `ruff check` and `pytest` on PRs
- Start with lenient rules and tighten over time

---

## Priority 4: Set Up a Real Testing Framework

**Effort:** Small

**Problem:** The `src/test/` files are standalone scripts and notebooks, not actual test suites. There is no `pytest`, no `unittest`, and no test runner configuration.

**Fix:**
- Add `pytest` to `requirements.txt`
- Create a `tests/` directory at the repo root
- Write starter tests for the most critical utilities (connectivity, color conversions, graph utils)
- Add a `pytest` section to `pyproject.toml` for configuration
- Wire tests into the CI pipeline from Priority 3

---

## Priority 5: Expand the README

**Effort:** Small

**Problem:** The README is 5 lines and provides no onboarding guidance. The `docs/` folder has general Python/git guides but nothing project-specific.

**Fix:** Add to `README.md`:
- Project architecture overview (what each `src/` subdirectory does)
- Step-by-step environment setup (referencing `.env_sample`)
- How to run the analysis notebooks
- Brief explanation of the data pipeline (NeuPrint -> processing -> results)
- Contributing guidelines or a pointer to them

---

## Priority 6: Break Up Monolithic Scripts

**Effort:** Large

**Problem:** Several files are enormous with mixed concerns:
- `contrib_plot_mip.py` -- 3,474 lines
- `contrib_plot.py` -- 3,794 lines
- `retinotopy_cb.py` -- 3,589 lines
- `resolution_cb.py` -- 2,907 lines

These files mix data loading, computation, and visualization, making them hard to navigate, test, or reuse.

**Fix:**
- Separate reusable computation from analysis-specific plotting
- Extract common patterns into focused modules
- Aim for files under ~500 lines where practical
- Keep notebook-specific orchestration in notebooks; move logic into importable modules

---

## Priority 7: Use Git LFS for Large Data Files

**Effort:** Small

**Problem:** `data/flywire/*.csv.gz` files (~50 MB total) are tracked directly in git, bloating the repo and slowing clones.

**Fix:**
- Install and configure Git LFS
- Migrate `data/flywire/*.csv.gz` and any other large binary files to LFS
- Keep small config files (like `params/*.xlsx`) in regular git

---

## Priority 8: Clean Up Code Debt

**Effort:** Medium

**Problem:** Accumulated technical debt across the codebase:
- 54+ unresolved TODO/FIXME comments
- 37+ DEBUG comments (especially in `quan_propagation/`)
- Potential bug: `ol_color.py:160` -- `return self.hex[which]` may cause infinite recursion in the `hex` property
- Typo: `ol_types.py:49` -- "Parmeters" should be "Parameters"
- Dead code: `helper.py:50-51` -- `df['color'].astype(dtype='object')` result is never assigned

**Fix:**
- Triage TODOs: fix, convert to GitHub issues, or remove
- Delete debug/commented-out code
- Fix the identified bugs and typos
- Add type hints consistently to `src/utils/` (imported everywhere), then expand outward
- Expand `.gitignore` to cover `.vscode/`, `.idea/`, `*.pyc`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, etc.
