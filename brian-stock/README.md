# BRIAN STOCK

Modular Streamlit investment research dashboard.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Secrets
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "YOUR_KEY"
```

## Structure
- `data/` market and news
- `analysis/` technical + quant/ML
- `ai/` AI provider/chat/client-message
- `portfolio/` portfolio calculations/storage
- `components/` UI/theme/charts
- `pages/` Streamlit pages
- `assets/` BRIAN STOCK logo/background
