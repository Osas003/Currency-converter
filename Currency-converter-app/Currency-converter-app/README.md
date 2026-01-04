Currency Converter with Historical Rates and Graphs

This is a small Streamlit app that converts currencies and shows historical exchange-rate graphs using exchangerate.host API.

Quick start

1. Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or cmd
.\.venv\Scripts\activate.bat
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. (Optional) Set up API key for full access

To use the full currency list and avoid rate limits, get a free API key from [exchangerate.host](https://exchangerate.host/):

- Create an account and get your free API key
- Open `.streamlit/secrets.toml` in the project folder
- Uncomment and add your key:
  ```toml
  EXCHANGERATE_API_KEY = "your_api_key_here"
  ```

Alternatively, set the environment variable:

```bash
set EXCHANGERATE_API_KEY=your_api_key_here
```

4. Run the app

```bash
streamlit run app.py
```

Features

- **Real-time currency conversion** with the latest exchange rates
- **Historical rate charts** showing trends over time (requires API key for best results)
- **Interactive graphs** using Plotly or Altair
- **Fallback mode** — if no API key, still works with a built-in list of popular currencies

Notes

- The app uses https://exchangerate.host which is free and doesn't require an API key, but offering one gives you access to more currencies and better rate limits.
- If you see "missing_access_key" error, you'll be offered the fallback currency list.
