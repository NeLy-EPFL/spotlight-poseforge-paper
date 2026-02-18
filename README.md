Analysis and visualization code for Wang-Chen et al. (2026).

## Installation
Prerequisite: install [uv](https://docs.astral.sh/uv/). Then,

```sh
git clone https://github.com/NeLy-EPFL/spotlight-poseforge-paper.git
cd spotlight-poseforge-paper/
uv sync --extra dev
uv run nbstripout --install --attributes .gitattributes
```

The `nbstripout` dependency under `dev` will remove outputs from `.ipynb` notebooks before committing to git (this avoid version-controlling to much binary data). To explicitly upload notebook outputs:
```sh
uv run nbstripout --uninstall
git add my_notebook.ipynb
git commit -m "my commit message"
uv run nbstripout --install --attributes .gitattributes
```