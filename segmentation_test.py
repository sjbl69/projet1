import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor

# 🔹 Chemin vers ton image
image_path = "assets/IMG/image_0.png"  # modifie ce chemin si besoin

# 🔹 Modèle SAM (Segment Anything)
model_type = "vit_b"  # petit modèle, rapide pour tests
sam = sam_model_registry[model_type](checkpoint="sam_vit_b_01ec64.pth")
sam.eval()

predictor = SamPredictor(sam)

# 🔹 Lire l'image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Impossible de trouver l'image : {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 🔹 Préparer le modèle pour l'image
predictor.set_image(image_rgb)

# 🔹 Segmentation automatique
masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    multimask_output=True
)

# 🔹 Sauvegarder le masque principal
mask = masks[0].astype("uint8") * 255
cv2.imwrite("resultat.png", mask)

print("✅ Segmentation terminée, résultat enregistré dans 'resultat.png'")
