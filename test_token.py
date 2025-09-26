import os
import requests
from dotenv import load_dotenv

# Charger le .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Vérifier qu’on a bien le token
if not HF_TOKEN:
    raise ValueError("Token introuvable ! Vérifie ton fichier .env")

# Faire une requête simple au modèle segformer
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/mattmdjaga/segformer_b2_clothes"

response = requests.post(API_URL, headers=headers, data=b"")
print("Code HTTP:", response.status_code)
print("Réponse:", response.json())
