# Setup

This repo's dataset work happens under `Dataset/` and Python work must use the `WEBUI` conda environment only.

## One-Time Environment Setup

The following commands were already run successfully in `C:\WebUIDetection\Dataset`:

```bash
npm install playwright
npm install @playwright/test
npx playwright install chromium
conda create --name WEBUI python=3.11
conda activate WEBUI
pip install pycocotools pyyaml pillow
```

## Working Rules

- Always activate the `WEBUI` conda environment before any Python command, notebook task, or package install.
- Do not create or use `venv` for this repository.
- Use Jupyter notebooks for training and testing workflows; do not build the dataset workflow around standalone `.py` scripts.
- Keep crawler outputs, URL tables, screenshots, annotations, and model artifacts under `Dataset/`.

## Recommended Notebook Setup

If the `WEBUI` environment is not already available as a Jupyter kernel, register it before starting notebook work:

```bash
conda activate WEBUI
pip install ipykernel
python -m ipykernel install --user --name WEBUI --display-name "Python (WEBUI)"
```

## Dataset Workspace Layout

- `Dataset/url-sources/` for scraped and deduplicated URL tables
- `Dataset/raw/screenshots/` for raw captures
- `Dataset/raw/annotations/` for intermediate annotations
- `Dataset/output/` for final images, labels, and COCO exports

## Notes

- The Playwright browser install is Chromium-only for now.
- Any future Python dependency changes should be installed into `WEBUI` and recorded here.