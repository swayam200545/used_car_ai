"""
Car Image Analysis & Visual Condition Assessment.

Analyzes uploaded car photographs using Computer Vision & Image Processing (PIL + NumPy):
  - Validates image dimensions, formats and clarity (Laplacian sharpness).
  - Assesses paint luster, contrast, reflection, and cosmetic wear.
  - Automatically classifies the photo viewpoint (Front/Exterior, Cabin/Dashboard, Wheel/Tyre, Engine/Detail).
  - Calculates a Visual Condition Score (1 to 10) and defect penalty modifier.
  - Generates lightweight base64 thumbnails for preview and MongoDB persistence.
"""
import io
import base64
import numpy as np
from PIL import Image, ImageStat, ImageFilter


def analyze_car_image(image_bytes, filename="car.jpg"):
    """
    Analyzes a single car photograph.
    Returns:
      {
        "filename": str,
        "width": int,
        "height": int,
        "sharpness": float,
        "brightness": float,
        "contrast": float,
        "viewpoint": str,
        "condition_rating": float (1-10),
        "condition_label": str,
        "tags": list[str],
        "thumbnail_b64": str (data URL),
      }
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        # Re-open after verify
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image file: {str(e)}", "filename": filename}

    width, height = img.size
    
    # Resize for fast, standardized analysis
    thumb_analysis = img.copy()
    thumb_analysis.thumbnail((600, 450), Image.Resampling.LANCZOS)
    arr = np.array(thumb_analysis, dtype=np.float32)

    # Grayscale for sharpness & edge analysis
    gray = thumb_analysis.convert("L")
    gray_arr = np.array(gray, dtype=np.float32)

    # 1. Sharpness via discrete Laplacian approximation
    # Kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    laplacian = (
        np.roll(gray_arr, 1, axis=0) +
        np.roll(gray_arr, -1, axis=0) +
        np.roll(gray_arr, 1, axis=1) +
        np.roll(gray_arr, -1, axis=1) -
        4 * gray_arr
    )
    sharpness_var = float(np.var(laplacian))

    # 2. Brightness & Contrast
    stat = ImageStat.Stat(thumb_analysis)
    brightness = float(np.mean(stat.mean))  # 0 to 255
    contrast = float(np.mean(stat.stddev))  # Standard deviation of color channels

    # 3. Color saturation and luster analysis
    hsv = thumb_analysis.convert("HSV")
    hsv_arr = np.array(hsv, dtype=np.float32)
    saturation = float(np.mean(hsv_arr[:, :, 1]))  # 0 to 255

    # 4. Viewpoint classification heuristic
    # Cabin/interior has darker average tones with high local contrast (steering, dashboard, vents)
    # Tyres have very low brightness and circular edge distributions
    # Exterior photos typically have high brightness, clear horizon or background variance
    tags = []
    viewpoint = "Exterior View"
    if brightness < 85 and contrast < 45:
        viewpoint = "Wheel & Tyre / Underbody"
        tags.append("Wheel/Underbody")
    elif brightness < 110 and contrast >= 45:
        viewpoint = "Cabin & Dashboard"
        tags.append("Interior Cabin")
    elif 110 <= brightness <= 190:
        viewpoint = "Exterior Body Angle"
        tags.append("Body Panel")
    else:
        viewpoint = "Showroom / Bright Exterior"
        tags.append("Bright Lighting")

    # Evaluate paint luster and clarity
    if sharpness_var > 220:
        tags.append("High Clarity")
    elif sharpness_var < 50:
        tags.append("Low Resolution / Blur")

    if contrast > 55 and saturation > 40:
        tags.append("Good Paint Luster")
    elif contrast < 35:
        tags.append("Dull / Flat Paint")

    # 5. Compute Condition Rating (1.0 to 10.0)
    # Baseline condition: 8.5 (Clean)
    score = 8.5
    # Penalize low sharpness/blur
    if sharpness_var < 45:
        score -= 1.0
    elif sharpness_var > 180:
        score += 0.5

    # Check contrast and color depth
    if contrast < 30:
        score -= 1.2
        tags.append("Visible Wear / Fading")
    elif contrast > 60:
        score += 0.5

    # Check if severely under- or over-exposed
    if brightness < 50 or brightness > 220:
        score -= 0.8
        tags.append("Poor Lighting Angle")

    score = round(float(np.clip(score, 4.0, 9.8)), 1)

    if score >= 9.0:
        label = "Showroom / Mint"
    elif score >= 8.0:
        label = "Clean / Excellent"
    elif score >= 6.5:
        label = "Average / Minor Scratches"
    else:
        label = "Moderate Wear / Attention Needed"

    # 6. Generate lightweight compressed thumbnail data URL (JPEG 80% quality, max 400x300)
    thumb = img.copy()
    thumb.thumbnail((400, 300), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80, optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    thumbnail_b64 = f"data:image/jpeg;base64,{b64_str}"

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "sharpness": round(sharpness_var, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "viewpoint": viewpoint,
        "condition_rating": score,
        "condition_label": label,
        "tags": tags,
        "thumbnail_b64": thumbnail_b64,
    }


def aggregate_images_analysis(images_results):
    """
    Summarizes multiple car photos into a unified condition profile.
    """
    if not images_results:
        return {
            "count": 0,
            "overall_condition_score": 8.0,
            "overall_condition_label": "Standard Market Average",
            "condition_adjustment_pct": 0.0,
            "viewpoints_covered": [],
            "identified_highlights": ["No photos uploaded - baseline model estimate used."],
            "gallery": [],
        }

    valid = [r for r in images_results if "error" not in r]
    if not valid:
        return {
            "count": 0,
            "overall_condition_score": 7.5,
            "overall_condition_label": "Upload Unreadable",
            "condition_adjustment_pct": 0.0,
            "viewpoints_covered": [],
            "identified_highlights": ["Uploaded files could not be processed."],
            "gallery": [],
        }

    ratings = [v["condition_rating"] for v in valid]
    avg_score = round(float(np.mean(ratings)), 1)
    viewpoints = list(set(v["viewpoint"] for v in valid))

    # Condition adjustment factor for pricing:
    # 9.0+ -> +3% to +5% (Well preserved)
    # 8.0 - 8.9 -> 0% to +2%
    # 7.0 - 7.9 -> -2% to -4% (Minor blemishes)
    # < 7.0 -> -6% to -10% (Visible scratches/wear)
    adj_pct = round((avg_score - 8.0) * 2.5, 1)
    adj_pct = float(np.clip(adj_pct, -10.0, 5.0))

    if avg_score >= 9.0:
        label = "Pristine / Showroom Condition"
    elif avg_score >= 8.0:
        label = "Clean & Well-Maintained"
    elif avg_score >= 7.0:
        label = "Fair / Minor Cosmetic Blemishes"
    else:
        label = "Visible Wear & Panel Scratches"

    highlights = []
    highlights.append(f"Processed {len(valid)} car photograph(s) covering {len(viewpoints)} angle(s).")
    if avg_score >= 8.5:
        highlights.append("Paint luster and body panels show healthy reflection and minimal surface degradation.")
    elif avg_score < 7.5:
        highlights.append("Visual analysis indicates faded luster or uneven panel contrast; inspect closely for touch-ups.")

    return {
        "count": len(valid),
        "overall_condition_score": avg_score,
        "overall_condition_label": label,
        "condition_adjustment_pct": adj_pct,
        "viewpoints_covered": viewpoints,
        "identified_highlights": highlights,
        "gallery": [
            {
                "filename": v["filename"],
                "viewpoint": v["viewpoint"],
                "rating": v["condition_rating"],
                "label": v["condition_label"],
                "tags": v["tags"],
                "thumbnail": v["thumbnail_b64"],
            }
            for v in valid
        ],
    }
