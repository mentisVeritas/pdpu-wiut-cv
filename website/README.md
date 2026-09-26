# Team website

Static site for GitHub Pages (workflow deploys the `website/` folder).

```bash
python3 -m http.server 8080 --directory website
```

Live demo on this page: upload a video + events JSON, or click **Load example events**, then click a bar to seek.

Model inference (CPU ok):

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

After Pages is enabled the public URL is:

`https://mentisveritas.github.io/pdpu-wiut-cv/`
