import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# --------------------------
# Paramètres
# --------------------------
image_folder = "images/test"             # dossier contenant les images de test
visualization_folder = "visualizations"  # dossier pour sauvegarder les résultats
checkpoint_path = "sam_vit_b_01ec64.pth" # chemin vers le checkpoint SAM
model_type = "vit_b"                     # modèle SAM

# Créer le dossier visualizations s'il n'existe pas
os.makedirs(visualization_folder, exist_ok=True)

# --------------------------
# Chargement du modèle SAM
# --------------------------
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.eval()
predictor = SamPredictor(sam)

# --------------------------
# Parcours des images
# --------------------------
for filename in os.listdir(image_folder):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    image_path = os.path.join(image_folder, filename)
    original = cv2.imread(image_path)
    if original is None:
        print(f"⚠️ Impossible de lire {filename}")
        continue

    image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # Segmentation automatique
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        multimask_output=True
    )

    # Masque principal
    mask = masks[0].astype("uint8") * 255

    # Sauvegarde du masque
    mask_path = os.path.join(visualization_folder, f"mask_{filename}")
    cv2.imwrite(mask_path, mask)

    # Création overlay avec code couleur (rouge)
    colored_mask = np.zeros_like(original)
    colored_mask[:, :, 2] = mask  # rouge
    overlay = cv2.addWeighted(original, 0.7, colored_mask, 0.3, 0)

    # Sauvegarde de l'overlay
    overlay_path = os.path.join(visualization_folder, f"overlay_{filename}")
    cv2.imwrite(overlay_path, overlay)

    # Affichage côte à côte
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Masque segmenté")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"✅ {filename} traité et visualisé")

print("🎉 Toutes les images ont été traitées et sauvegardées dans 'visualizations/'")
