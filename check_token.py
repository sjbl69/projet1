import os
import requests
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("❌ Aucun token trouvé dans .env")

# En-têtes avec le token
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# URL API pour vérifier le token
url = "https://huggingface.co/api/whoami-v2"

try:
    r = requests.get(url, headers=headers)
    r.raise_for_status()  # ⚠️ Lève une erreur si le code HTTP n'est pas 200
except requests.exceptions.RequestException as e:
    print(f"❌ Erreur lors de la requête : {e}")
else:
    print("✅ Code HTTP:", r.status_code)
    print("Réponse JSON:", r.json())
