import os
import requests
from dotenv import load_dotenv
import json

# Charger le token depuis .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/mattmdjaga/segformer_b2_clothes"

if HF_TOKEN is None:
    raise ValueError("❌ Token Hugging Face introuvable dans .env !")

def query(filename):
    """Envoie une image à l’API Hugging Face et récupère la réponse"""
    # Déterminer le type de l'image (ici PNG)
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/png"
    }
    
    with open(filename, "rb") as f:
        data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)

    print("📥 Code HTTP:", response.status_code)
    print("📥 Réponse brute (500 premiers caractères):")
    print(response.text[:500])

    try:
        return response.json()
    except Exception as e:
        print("⚠️ Erreur JSON:", e)
        return None

# -----------------------------
# Test avec une image de ton dossier
# -----------------------------
test_image = "assets/IMG/image_1.png"

print(f"📤 Envoi de {test_image} au modèle Hugging Face...")
result = query(test_image)

if result is None:
    print("⚠️ Impossible de décoder la réponse en JSON. Vérifie les logs ci-dessus.")
else:
    print("✅ Résultat JSON:")
    print(json.dumps(result, indent=2))

