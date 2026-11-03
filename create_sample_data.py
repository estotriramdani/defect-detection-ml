"""
Script untuk membuat sample data training untuk model deteksi cacat label
Menghasilkan gambar simulasi untuk kategori 'good' dan 'defect'
Author: Developer
Date: July 15, 2025
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont
import random
from datetime import datetime

# Pastikan output UTF-8 agar emoji tidak error di konsol Windows (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def create_directory_structure():
    """
    Membuat struktur direktori untuk training data
    """
    base_dir = "static/training_data"
    good_dir = os.path.join(base_dir, "good")
    defect_dir = os.path.join(base_dir, "defect")
    
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(defect_dir, exist_ok=True)
    
    return good_dir, defect_dir

def generate_label_background(width=400, height=300):
    """
    Generate background label yang realistis
    """
    # Warna background yang umum untuk label
    bg_colors = [
        (255, 255, 255),  # Putih
        (245, 245, 245),  # Abu-abu terang
        (255, 248, 220),  # Krem
        (230, 230, 250),  # Lavender terang
        (240, 248, 255)   # Alice blue
    ]
    
    bg_color = random.choice(bg_colors)
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Tambahkan noise halus untuk membuat lebih realistis
    for _ in range(random.randint(50, 100)):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        noise_color = tuple(min(255, max(0, c + random.randint(-10, 10))) for c in bg_color)
        draw.point((x, y), noise_color)
    
    return img, draw

def add_text_elements(img, draw, is_good=True):
    """
    Menambahkan elemen teks pada label
    """
    width, height = img.size
    
    # Simulasi teks yang biasa ada di label
    texts = [
        "PRODUCT LABEL",
        "Batch: 2025-001",
        "Exp: 2026-07-15",
        "Made in Indonesia",
        "Net Wt: 500g"
    ]
    
    try:
        # Coba gunakan font sistem
        font_size = random.randint(12, 20)
        # Fallback ke font default jika tidak ada
        font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    for i, text in enumerate(texts):
        x = random.randint(10, width // 4)
        y = 20 + i * 30
        
        if is_good:
            # Teks normal untuk label good
            text_color = (0, 0, 0)  # Hitam
        else:
            # Kadang teks blur atau tidak jelas untuk defect
            if random.random() < 0.3:
                text_color = (128, 128, 128)  # Abu-abu (blur effect)
            else:
                text_color = (0, 0, 0)
        
        draw.text((x, y), text, fill=text_color, font=font)

def add_barcode_simulation(img, draw, is_good=True):
    """
    Menambahkan simulasi barcode
    """
    width, height = img.size
    
    # Posisi barcode
    barcode_x = width - 150
    barcode_y = height - 100
    barcode_width = 120
    barcode_height = 60
    
    if is_good:
        # Barcode normal
        bar_color = (0, 0, 0)
        for i in range(0, barcode_width, 3):
            if random.random() < 0.6:  # 60% chance untuk bar
                x1 = barcode_x + i
                x2 = barcode_x + i + random.randint(1, 2)
                draw.rectangle([x1, barcode_y, x2, barcode_y + barcode_height], fill=bar_color)
    else:
        # Barcode rusak untuk defect
        if random.random() < 0.7:  # 70% chance ada masalah barcode
            bar_color = (100, 100, 100)  # Warna pudar
            for i in range(0, barcode_width, 3):
                if random.random() < 0.4:  # Lebih sedikit bar
                    x1 = barcode_x + i
                    x2 = barcode_x + i + random.randint(1, 3)
                    draw.rectangle([x1, barcode_y, x2, barcode_y + barcode_height], fill=bar_color)

def add_defects(img, draw):
    """
    Menambahkan berbagai jenis cacat pada label
    """
    width, height = img.size
    defect_types = ['scratch', 'stain', 'tear', 'wrinkle', 'fade']
    num_defects = random.randint(1, 3)
    
    for _ in range(num_defects):
        defect_type = random.choice(defect_types)
        
        if defect_type == 'scratch':
            # Goresan
            x1 = random.randint(0, width-50)
            y1 = random.randint(0, height-10)
            x2 = x1 + random.randint(20, 100)
            y2 = y1 + random.randint(-5, 5)
            draw.line([x1, y1, x2, y2], fill=(80, 80, 80), width=random.randint(1, 3))
            
        elif defect_type == 'stain':
            # Noda
            cx = random.randint(20, width-20)
            cy = random.randint(20, height-20)
            radius = random.randint(5, 20)
            stain_color = (random.randint(100, 200), random.randint(100, 150), random.randint(100, 150))
            draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=stain_color)
            
        elif defect_type == 'tear':
            # Sobek
            x = random.randint(10, width-10)
            y = random.randint(10, height-10)
            points = [(x, y), (x+random.randint(-10, 10), y+random.randint(10, 30)), 
                     (x+random.randint(-5, 15), y+random.randint(20, 40))]
            draw.polygon(points, fill=(200, 200, 200))
            
        elif defect_type == 'wrinkle':
            # Kerutan
            for _ in range(random.randint(2, 5)):
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                x2 = x1 + random.randint(-20, 20)
                y2 = y1 + random.randint(-20, 20)
                draw.line([x1, y1, x2, y2], fill=(180, 180, 180), width=1)
                
        elif defect_type == 'fade':
            # Warna pudar
            fade_area = [
                random.randint(0, width//2),
                random.randint(0, height//2),
                random.randint(width//2, width),
                random.randint(height//2, height)
            ]
            overlay = Image.new('RGB', (width, height), (220, 220, 220))
            img.paste(overlay, (0, 0), Image.new('L', (width, height), 50))

def add_good_quality_marks(img, draw):
    """
    Menambahkan tanda kualitas baik (untuk label good)
    """
    width, height = img.size
    
    # Tambahkan border yang rapi
    border_color = (100, 100, 100)
    draw.rectangle([2, 2, width-3, height-3], outline=border_color, width=2)
    
    # Tambahkan logo/seal simulasi
    seal_x = width - 80
    seal_y = 10
    seal_radius = 25
    draw.ellipse([seal_x-seal_radius, seal_y, seal_x+seal_radius, seal_y+2*seal_radius], 
                outline=(0, 100, 0), width=2)
    draw.text((seal_x-15, seal_y+seal_radius-5), "QC", fill=(0, 100, 0))

def generate_good_sample(filename, good_dir):
    """
    Generate sample gambar label yang baik
    """
    img, draw = generate_label_background()
    add_text_elements(img, draw, is_good=True)
    add_barcode_simulation(img, draw, is_good=True)
    add_good_quality_marks(img, draw)
    
    # Simpan gambar
    filepath = os.path.join(good_dir, filename)
    img.save(filepath, 'PNG', quality=90)
    print(f"Generated good sample: {filename}")

def generate_defect_sample(filename, defect_dir):
    """
    Generate sample gambar label yang cacat
    """
    img, draw = generate_label_background()
    add_text_elements(img, draw, is_good=False)
    add_barcode_simulation(img, draw, is_good=False)
    add_defects(img, draw)
    
    # Simpan gambar
    filepath = os.path.join(defect_dir, filename)
    img.save(filepath, 'PNG', quality=70)  # Kualitas lebih rendah untuk defect
    print(f"Generated defect sample: {filename}")

def main():
    """
    Main function untuk generate sample data
    """
    print("🎯 Membuat sample data training...")
    
    # Buat struktur direktori
    good_dir, defect_dir = create_directory_structure()
    
    # Generate timestamp untuk nama file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate good samples
    print("\n📸 Generating GOOD samples...")
    for i in range(1, 16):  # 15 sample good
        filename = f"sample_good_{i:03d}_{timestamp}.png"
        generate_good_sample(filename, good_dir)
    
    # Generate defect samples  
    print("\n⚠️  Generating DEFECT samples...")
    for i in range(1, 16):  # 15 sample defect
        filename = f"sample_defect_{i:03d}_{timestamp}.png"
        generate_defect_sample(filename, defect_dir)
    
    print(f"\n✅ Sample data berhasil dibuat!")
    print(f"   📁 Good samples: {good_dir}")
    print(f"   📁 Defect samples: {defect_dir}")
    print(f"   📊 Total: 30 sample images (15 good + 15 defect)")

if __name__ == "__main__":
    main()
