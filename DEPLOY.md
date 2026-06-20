# Deploying Athena

Athena has two public-facing pieces:

1. **The live interactive app** — the Streamlit dashboard, hosted free on
   **Streamlit Community Cloud**. This runs the Python engine on their servers
   and gives you a public URL anyone can open in a browser (no install).
2. **The landing page** — `docs/index.html`, hosted free on **GitHub Pages**,
   with a "Launch live app" button that points at the Streamlit URL.

Do them in this order so you have the app URL before wiring up the button.

---

## 1. Deploy the live app (Streamlit Community Cloud)

1. Make sure the repo is pushed to GitHub (`main` branch).
2. Go to **https://share.streamlit.io** and sign in with your GitHub account.
3. Click **Create app → Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `dana307/athen-model`
   - **Branch:** `main`
   - **Main file path:** `dashboard/streamlit_app.py`
5. (Optional) **Advanced settings → Python version:** `3.11`.
6. Click **Deploy**. The first build installs `requirements.txt` and takes a few
   minutes. When it finishes you'll get a URL like
   `https://athen-model.streamlit.app` (you can set a custom subdomain in the
   app's settings).

**Notes**
- Live market data (yfinance) works from the cloud, but shared IPs can get
  rate-limited. The app automatically falls back to bundled **demo data**, and
  there's a *"Use demo data"* checkbox, so the deployed app always works.
- Free tier gives ~1 GB RAM — plenty for Athena's workloads.

## 2. Publish the landing page (GitHub Pages)

1. In the repo on GitHub, go to **Settings → Pages**.
2. Under **Build and deployment → Source**, choose **Deploy from a branch**.
3. Set **Branch:** `main` and **Folder:** `/docs`, then **Save**.
4. After ~1 minute your page is live at
   **https://dana307.github.io/athen-model/**

## 3. Wire the button to your app

Open `docs/index.html`, find this line near the top of the hero:

```html
<a class="btn btn-primary" id="launch" href="https://YOUR-APP.streamlit.app" ...>
```

Replace `https://YOUR-APP.streamlit.app` with your real Streamlit URL from
step 1, then commit and push:

```bash
git add docs/index.html && git commit -m "Point landing page at live app" && git push
```

That's it — share the GitHub Pages link, and anyone can read the overview and
launch the live, runnable app from their browser.
