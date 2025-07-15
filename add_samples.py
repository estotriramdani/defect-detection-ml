#!/usr/bin/env python3

import os
from PIL import Image, ImageDraw
import random

# Buat 5 gambar defect tambahan
defect_folder = "static/training_data/defect"
os.makedirs(defect_folder, exist_ok=True)

for i in range(5):
    # Buat gambar 224x224 putih
    img = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img)
    
    # Tambahkan defect berupa noda hitam
    for j in range(random.randint(3, 8)):
        x = random.randint(20, 200)
        y = random.randint(20, 200)
        r = random.randint(5, 20)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(50, 50, 50))
    
    # Simpan
    filename = f"manual_defect_{i+1:02d}.png"
    filepath = os.path.join(defect_folder, filename)
    img.save(filepath)
    print(f"Created: {filename}")

print("✅ Done creating additional defect samples!")
