"""
tumor_classifier.py
===================
Tumor characterization module for segmented liver lesions.
Computes morphological, geometric, and intensity features from predicted masks:
- Lesion Area (pixels & mm²)
- Shape Irregularity / Solidity (contour area / convex hull area)
- Edge Roughness (perimeter / hull perimeter)
- Lesion Count (number of disconnected components)
- Circularity / Heterogeneity

Applies clinical decision rules to classify lesions as:
- 'Benign (Non-Cancerous)' or
- 'Malignant (Cancerous)'
with calibrated confidence % and clear medical disclaimers.
"""

import cv2
import numpy as np


DISCLAIMER_TEXT = "Demo estimate - not a medical diagnosis. For research & educational purposes only."


def extract_morphological_features(mask, mri_img=None, pixel_spacing_mm=1.25):
    """
    Extracts geometric and textural features from a binary mask (and optional MRI).
    
    Args:
        mask: 2D numpy array (uint8 or bool, 256x256).
        mri_img: Optional 2D numpy array (MRI grayscale image).
        pixel_spacing_mm: MRI in-plane spatial resolution in mm/pixel.
        
    Returns:
        Dictionary of extracted quantitative features.
    """
    # Ensure binary uint8
    if mask.dtype != np.uint8:
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
    else:
        binary_mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    total_pixels = int(np.count_nonzero(binary_mask))
    pixel_area_mm2 = (pixel_spacing_mm ** 2)
    area_mm2 = round(total_pixels * pixel_area_mm2, 1)

    if total_pixels == 0:
        return {
            "has_lesion": False,
            "total_pixels": 0,
            "area_mm2": 0.0,
            "lesion_count": 0,
            "solidity": 1.0,
            "circularity": 1.0,
            "roughness": 1.0,
            "intensity_std": 0.0,
            "contours": []
        }

    # Find contours of lesions
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter noise (< 10 pixels)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= 10]
    lesion_count = len(valid_contours)

    if lesion_count == 0:
        return {
            "has_lesion": False,
            "total_pixels": total_pixels,
            "area_mm2": area_mm2,
            "lesion_count": 0,
            "solidity": 1.0,
            "circularity": 1.0,
            "roughness": 1.0,
            "intensity_std": 0.0,
            "contours": []
        }

    # Analyze primary (largest) lesion
    largest_contour = max(valid_contours, key=cv2.contourArea)
    primary_area = float(cv2.contourArea(largest_contour))
    primary_perimeter = float(cv2.arcLength(largest_contour, True))

    # Convex Hull & Solidity
    hull = cv2.convexHull(largest_contour)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))

    solidity = (primary_area / hull_area) if hull_area > 0 else 1.0
    solidity = min(1.0, max(0.0, solidity))

    # Circularity (Isoperimetric Quotient: 4*pi*Area / P^2)
    circularity = (4.0 * np.pi * primary_area) / (primary_perimeter ** 2) if primary_perimeter > 0 else 1.0
    circularity = min(1.0, max(0.0, circularity))

    # Edge roughness: ratio of actual perimeter to convex hull perimeter
    roughness = (primary_perimeter / hull_perimeter) if hull_perimeter > 0 else 1.0
    roughness = max(1.0, roughness)

    # Intensity heterogeneity (if MRI image provided)
    intensity_std = 0.0
    if mri_img is not None:
        lesion_pixels = mri_img[binary_mask > 0]
        if len(lesion_pixels) > 0:
            intensity_std = float(np.std(lesion_pixels))
            # Normalize if 0-255 vs 0-1
            if mri_img.max() > 1.0:
                intensity_std /= 255.0

    return {
        "has_lesion": True,
        "total_pixels": total_pixels,
        "area_mm2": area_mm2,
        "lesion_count": lesion_count,
        "solidity": round(solidity, 4),
        "circularity": round(circularity, 4),
        "roughness": round(roughness, 4),
        "intensity_std": round(intensity_std, 4),
        "primary_area": round(primary_area, 1),
        "primary_perimeter": round(primary_perimeter, 1)
    }


def classify_tumor_features(features):
    """
    Classifies segmented liver lesion based on clinical morphology:
    - Benign: smooth, regular, well-circumscribed, high solidity, solitary.
    - Malignant: irregular margins, low solidity, high roughness, multi-focal, large.
    
    Returns structured diagnosis result with confidence % and medical disclaimer.
    """
    if not features["has_lesion"] or features["total_pixels"] < 25:
        return {
            "prediction": "Normal / No Lesion",
            "type_code": "normal",
            "confidence": 99.0,
            "badge_class": "badge-normal",
            "summary": "No suspicious hepatic lesion detected. Liver parenchyma shows normal homogeneity.",
            "reasons": [
                "Zero or negligible hyper/hypointense lesion volume detected",
                "Segmentation mask area is below diagnostic lesion threshold"
            ],
            "disclaimer": DISCLAIMER_TEXT
        }

    # Score malignancy risk (0 to 100)
    malignancy_score = 0
    reasons = []

    # 1. Shape Irregularity (Solidity): Infiltrative/lobulated margins have low solidity
    solidity = features["solidity"]
    if solidity < 0.80:
        malignancy_score += 30
        reasons.append(f"Infiltrative/jagged lesion contour (Solidity: {solidity:.2f} < 0.80)")
    elif solidity < 0.88:
        malignancy_score += 15
        reasons.append(f"Mildly lobulated contour (Solidity: {solidity:.2f})")
    else:
        reasons.append(f"Smooth, circumscribed margin (Solidity: {solidity:.2f} >= 0.88)")

    # 2. Edge Roughness / Spiculation
    roughness = features["roughness"]
    if roughness > 1.25:
        malignancy_score += 25
        reasons.append(f"Significant border spiculation (Edge roughness index: {roughness:.2f})")
    elif roughness > 1.12:
        malignancy_score += 10
        reasons.append(f"Minor border irregularity (Edge roughness index: {roughness:.2f})")
    else:
        reasons.append(f"Regular, non-spiculated perimeter (Roughness: {roughness:.2f})")

    # 3. Lesion Multiplicity (Satellite nodules / multifocal)
    count = features["lesion_count"]
    if count >= 3:
        malignancy_score += 25
        reasons.append(f"Multiple discrete lesions / satellite foci ({count} nodules)")
    elif count == 2:
        malignancy_score += 15
        reasons.append(f"Bifocal hepatic lesion pattern ({count} nodules)")
    else:
        reasons.append("Solitary focal lesion")

    # 4. Total Volume / Pixel Area
    area = features["total_pixels"]
    if area > 1400:
        malignancy_score += 15
        reasons.append(f"Large lesion mass ({features['area_mm2']} mm², {area} px)")
    elif area > 800:
        malignancy_score += 8

    # 5. Internal Texture Heterogeneity
    int_std = features["intensity_std"]
    if int_std > 0.08:
        malignancy_score += 10
        reasons.append(f"Heterogeneous internal signal (Intensity variance: {int_std:.3f})")

    # Classification threshold
    if malignancy_score >= 45:
        # Malignant
        # Map score 45-100 to 72% - 98%
        confidence = 72.0 + (malignancy_score - 45) * (26.0 / 55.0)
        confidence = min(98.5, round(confidence, 1))
        
        return {
            "prediction": "Malignant (Cancerous)",
            "type_code": "malignant",
            "confidence": confidence,
            "badge_class": "badge-malignant",
            "malignancy_score": malignancy_score,
            "summary": "Morphological features demonstrate high-risk characteristics suggestive of Hepatocellular Carcinoma (HCC) or Metastatic Neoplasm.",
            "reasons": reasons,
            "disclaimer": DISCLAIMER_TEXT
        }
    else:
        # Benign
        # Map score 0-44 to 75% - 99%
        confidence = 99.0 - malignancy_score * (24.0 / 45.0)
        confidence = min(99.0, max(75.0, round(confidence, 1)))

        return {
            "prediction": "Benign (Non-Cancerous)",
            "type_code": "benign",
            "confidence": confidence,
            "badge_class": "badge-benign",
            "malignancy_score": malignancy_score,
            "summary": "Morphological features demonstrate low-risk characteristics consistent with Benign Hepatic Hemangioma or Simple Liver Cyst.",
            "reasons": reasons,
            "disclaimer": DISCLAIMER_TEXT
        }
