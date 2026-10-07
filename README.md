# Who's a LinkedIn user?

Streamlit app for Part 2 of the Programming II final project. Describe a customer (income, education, age, gender, married, parent) and the logistic regression model from Part 1 says whether that person is likely on LinkedIn and how likely. Same data cleaning, same 80/20 split, same seed as the notebook, so the numbers match.

What's in the app:

- **Headline** - the classification (user / not a user), the probability, and a meter that shows it against the survey average and the 50% cutoff
- **This person** - the probability curve across every age for the current profile, and a "change one thing" chart showing what one step in each input does
- **Segments and platforms** - a heatmap of where LinkedIn users are concentrated (age x income or age x education) with the current person's cell outlined, and what else that segment uses across all 11 platforms in the survey
- **About the model** - accuracy, recall, precision and base rate on the test set, the odds ratios, and the confusion matrix

## Files

- `app.py` - the app
- `social_media_usage.csv` - the survey data the model trains on. Has to sit next to `app.py`
- `requirements.txt` - packages
- `.streamlit/config.toml` - just the accent color for the controls
- `DEMO_SCRIPT.md` - talking points for the recording

## Run it locally

```
cd linkedin_app
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. Stop it with Ctrl+C in the terminal.

If you want a clean environment first:

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (bonus)

1. Make a new public repo on GitHub (e.g. `linkedin-predictor`).
2. From this folder:
   ```
   git init
   git add app.py requirements.txt social_media_usage.csv README.md .streamlit/config.toml
   git commit -m "LinkedIn user prediction app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/linkedin-predictor.git
   git push -u origin main
   ```
3. Go to https://share.streamlit.io, sign in with GitHub, click **Create app** > **Deploy a public app from GitHub**.
4. Pick the repo, branch `main`, main file `app.py`. Click **Deploy**.
5. It takes a minute or two to build. The URL will look like `https://<your-username>-linkedin-predictor-app-xxxx.streamlit.app`.

Any push to `main` redeploys automatically.
