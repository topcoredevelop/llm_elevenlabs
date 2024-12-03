# OpenAI API Proxy

En FastAPI-basert proxy for OpenAI's API med støtte for streaming, token-håndtering og automatisk modell-validering. Designet for enkel deployment på Railway.app.

## Funksjoner

- 🚀 Full streaming-støtte for ChatGPT-responser
- 📊 Automatisk token-håndtering og justering
- ✅ Validering av modeller og parametre
- 🔄 CORS-støtte for web-integrasjoner
- 🏥 Innebygd helsesjekk-endepunkt
- 📝 Detaljert logging
- 🐳 Docker-støtte

## API Endepunkter

- `POST /v1/chat/completions` - Hoved-endepunkt for chat completions
- `GET /health` - Helsesjekk-endepunkt
- `GET /` - API informasjon

## Oppsett

### Forutsetninger

- Python 3.9 eller nyere
- OpenAI API-nøkkel
- Git
- Docker (valgfritt)

### Miljøvariabler

```env
OPENAI_API_KEY=din-api-nøkkel-her
PORT=8000  # Valgfri, standard er 8000
```

### Lokal Kjøring

1. Klon repositoriet:
```bash
git clone [repo-url]
cd [repo-navn]
```

2. Installer avhengigheter:
```bash
pip install -r requirements.txt
```

3. Opprett `.env` fil med din OpenAI API-nøkkel:
```env
OPENAI_API_KEY=din-api-nøkkel-her
```

4. Kjør applikasjonen:
```bash
python main.py
```

### Docker Kjøring

1. Bygg Docker image:
```bash
docker build -t openai-proxy .
```

2. Kjør container:
```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=din-api-nøkkel-her openai-proxy
```

## Deployment på Railway

1. Fork eller push dette repositoriet til GitHub
2. Koble til Railway.app med GitHub
3. Velg repositoriet i Railway
4. Legg til miljøvariabel i Railway:
   - `OPENAI_API_KEY`: Din OpenAI API-nøkkel
5. Deploy!

## Bruk

### Eksempel på forespørsel

```python
import requests
import json

url = "din-railway-url/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": "Hei, hvordan har du det?"
        }
    ],
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=data)
print(json.dumps(response.json(), indent=2))
```

### Streaming Eksempel

```python
import requests

url = "din-railway-url/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": "Fortell meg en historie"
        }
    ],
    "stream": True
}

response = requests.post(url, headers=headers, json=data, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## Støttede Modeller

- gpt-4
- gpt-4-32k
- gpt-3.5-turbo
- gpt-3.5-turbo-16k

## Feilsøking

### Vanlige Feil

1. **"OpenAI API key not found"**
   - Sjekk at OPENAI_API_KEY er satt korrekt i miljøvariabler

2. **"Model not found"**
   - Verifiser at du bruker en støttet modell

3. **"Token limit exceeded"**
   - Reduser lengden på meldingene eller juster max_tokens

### Logging

Applikasjonen logger til stdout med detaljert informasjon om feil og forespørsler.

## Bidrag

Bidrag er velkomne! Vennligst følg disse stegene:

1. Fork repositoriet
2. Opprett en feature branch
3. Commit endringene dine
4. Push til branchen
5. Åpne en Pull Request

## Lisens

Dette prosjektet er lisensiert under MIT Lisens.
