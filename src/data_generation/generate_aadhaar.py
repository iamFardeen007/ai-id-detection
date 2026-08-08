# src/data_generation/generate_aadhaar.py
# Purpose: Generate synthetic Aadhaar-style cards with fake data.
# All data is FAKE via Faker library. No real UIDs. For research only.

from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import random
import os
from pathlib import Path

fake = Faker('en_IN')  # Indian locale for realistic names/addresses

CARD_W, CARD_H = 1011, 638  # Standard Aadhaar-like aspect ratio

def generate_fake_uid():
    """12-digit fake UID. Won't match real Verhoeff checksum = safe."""
    return f"{random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"

def get_fonts():
    """Try Mac system fonts, fall back to default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return (
                ImageFont.truetype(path, 28),
                ImageFont.truetype(path, 20),
                ImageFont.truetype(path, 16),
            )
    d = ImageFont.load_default()
    return d, d, d

def create_synthetic_aadhaar(output_path, sample_id):
    img = Image.new('RGB', (CARD_W, CARD_H), color=(255, 250, 240))
    draw = ImageDraw.Draw(img)
    font_large, font_med, font_small = get_fonts()

    # Top orange band
    draw.rectangle([(0, 0), (CARD_W, 80)], fill=(230, 126, 34))
    draw.text((250, 25), "GOVERNMENT OF INDIA (SYNTHETIC)", fill='white', font=font_large)

    # Photo placeholder
    draw.rectangle([(30, 120), (230, 380)], fill=(200, 200, 200), outline='black', width=2)
    draw.text((90, 240), "PHOTO", fill='black', font=font_med)

    # Fake personal details
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime('%d/%m/%Y')
    gender = random.choice(['MALE', 'FEMALE'])
    uid = generate_fake_uid()

    y = 130
    draw.text((260, y), f"Name: {name}", fill='black', font=font_med); y += 40
    draw.text((260, y), f"DOB: {dob}", fill='black', font=font_med); y += 40
    draw.text((260, y), f"Gender: {gender}", fill='black', font=font_med); y += 40
    draw.text((260, y), f"Address: {fake.address()[:50]}", fill='black', font=font_small); y += 40

    # UID footer band
    draw.rectangle([(0, 520), (CARD_W, 600)], fill=(230, 126, 34))
    draw.text((280, 545), uid, fill='white', font=font_large)

    # Legal safety watermark
    draw.text((400, 610), "SYNTHETIC - FOR RESEARCH ONLY", fill='red', font=font_small)

    img.save(output_path, 'JPEG', quality=random.randint(75, 95))
    return {'id': sample_id, 'name': name, 'dob': dob, 'gender': gender, 'uid': uid}

if __name__ == "__main__":
    output_dir = Path.home() / "ai_id_detection" / "data" / "raw" / "real_synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating 20 sample synthetic Aadhaars...")
    for i in range(20):
        path = output_dir / f"real_{i:04d}.jpg"
        create_synthetic_aadhaar(path, i)
        print(f"  ✓ Generated {path.name}")

    print(f"\n✅ Done! Check: {output_dir}")