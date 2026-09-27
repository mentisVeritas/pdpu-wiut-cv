# Team website

Static site served by GitHub Pages from this folder (`.github/workflows/pages.yml`).

- `index.html`, `styles.css`, `app.js`: the page; everything data-driven is read from `assets/`.
- `assets/`: built from the real pipeline output by
  `python scripts/build_site_assets.py --videos samples --preview samples/h264 --pred predictions_samples.json --out website/assets`
  (annotated clips, per-clip JSON with events / signal / counts / risk, heat maps, flow maps, one still per class).
- Live demo: the Hugging Face Space built by `scripts/pack_space.py`; set `DEMO_URL` in `app.js`.

Local preview: `python3 -m http.server 8080 --directory website`
