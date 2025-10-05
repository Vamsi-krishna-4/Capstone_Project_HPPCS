# demo_data.py
"""
Generate synthetic chest X-ray-like images and paired clinical notes.
All generated images are stored in the same directory (no sub-directories),
so evaluators can access them via "./demo_xray_1.png" etc.
"""

from PIL import Image, ImageDraw, ImageFilter
import random
import os

def create_demo_xray(path: str, width=512, height=512, seed=None):
    """
    Create a synthetic grayscale 'xray' style image and save to path.
    This is intentionally simple and synthetic for testing.
    """
    if seed is not None:
        random.seed(seed)
    img = Image.new("L", (width, height), color=30)
    draw = ImageDraw.Draw(img)

    # add circular opacities
    for _ in range(random.randint(3, 6)):
        r = random.randint(width//12, width//4)
        x = random.randint(r, width - r)
        y = random.randint(r, height - r)
        intensity = random.randint(60, 200)
        bbox = [x-r, y-r, x+r, y+r]
        draw.ellipse(bbox, fill=intensity)

    # add rib-like slightly curved lines
    for i in range(10):
        offset = int((i - 5) * 10)
        for j in range(4):
            start = (0, int(height*0.2 + offset + j*1.5))
            end = (width, int(height*0.8 + offset + j*1.4))
            draw.line([start, end], fill=random.randint(40, 75), width=1)

    img = img.filter(ImageFilter.GaussianBlur(radius=3))

    # convert to RGB for CLIP compatibility
    img_rgb = img.convert("RGB")
    img_rgb.save(path)
    return path

def generate_demo_pairs(n=5):
    """
    Generate n demo X-rays and return list of (image_path, clinical_note).
    Saves files to ./demo_xray_{i}.png
    """
    notes = [
        "45-year-old male with fever, productive cough and dyspnea. Suspect lobar pneumonia.",
        "60-year-old female with chronic cough and smoking history. Consider COPD exacerbation.",
        "30-year-old male after blunt chest trauma with pleuritic chest pain. Rule out pneumothorax.",
        "25-year-old female with sudden onset shortness of breath and pleuritic chest pain. Consider pulmonary embolism.",
        "70-year-old male with progressive dyspnea and lower-extremity edema. Consider congestive heart failure."
    ]
    pairs = []
    for i in range(1, n+1):
        path = f"./demo_xray_{i}.png"
        create_demo_xray(path, seed=i)
        note = notes[(i-1) % len(notes)]
        pairs.append((path, note))
    return pairs

if __name__ == "__main__":
    pairs = generate_demo_pairs()
    for p in pairs:
        print(p)
