"""
Script untuk membuat sample data training untuk model deteksi cacat label
Menghasilkan gambar simulasi untuk kategori 'good' dan 'defect'
Author: Developer
Date: July 15, 2025
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from datetime import datetime
import colorsys

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

def create_sample_good_image(width=224, height=224):
    """
    Membuat gambar simulasi label yang baik (good)
    
    Args:
        width (int): Lebar gambar
        height (int): Tinggi gambar
        
    Returns:
        PIL.Image: Gambar label yang baik
    """
    # Buat gambar dengan background putih
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Simulasi label dengan border yang rapi
    border_color = (0, 0, 0)  # Hitam
    border_width = 3
    
    # Gambar border luar
    draw.rectangle([10, 10, width-10, height-10], outline=border_color, width=border_width)
    
    # Simulasi teks label
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Teks produk
    text_lines = [
        "PRODUK BERKUALITAS",
        "Made in Indonesia",
        "Exp: 12/2025",
        "Net: 500g"
    ]
    
    y_pos = 30
    for line in text_lines:
        # Hitung posisi tengah
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x_pos = (width - text_width) // 2
        
        draw.text((x_pos, y_pos), line, fill=(0, 0, 0), font=font)
        y_pos += 35
    
    # Tambahkan logo simulasi (kotak kecil)
    logo_size = 30
    logo_x = (width - logo_size) // 2
    logo_y = height - logo_size - 20
    draw.rectangle([logo_x, logo_y, logo_x + logo_size, logo_y + logo_size], 
                  fill=(0, 100, 200), outline=(0, 0, 0), width=2)
    
    # Tambahkan sedikit noise untuk variasi
    noise_level = 5
    for _ in range(20):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        noise_color = (
            min(255, max(0, 255 + random.randint(-noise_level, noise_level))),
            min(255, max(0, 255 + random.randint(-noise_level, noise_level))),
            min(255, max(0, 255 + random.randint(-noise_level, noise_level)))
        )
        draw.point((x, y), fill=noise_color)
    
    return img

def create_sample_defect_image(width=224, height=224):
    """
    Membuat gambar simulasi label yang cacat (defect)
    
    Args:
        width (int): Lebar gambar
        height (int): Tinggi gambar
        
    Returns:
        PIL.Image: Gambar label yang cacat
    """
    # Buat gambar dengan background putih
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Defect type 1: Border tidak sempurna
    border_color = (0, 0, 0)
    border_width = random.choice([1, 2, 4, 5])  # Border tidak konsisten
    
    # Gambar border dengan masalah
    points = [
        [10 + random.randint(-3, 3), 10 + random.randint(-3, 3)],
        [width-10 + random.randint(-3, 3), 10 + random.randint(-3, 3)],
        [width-10 + random.randint(-3, 3), height-10 + random.randint(-3, 3)],
        [10 + random.randint(-3, 3), height-10 + random.randint(-3, 3)]
    ]
    
    # Border tidak rapi
    for i in range(len(points)):
        start = points[i]
        end = points[(i + 1) % len(points)]
        draw.line(start + end, fill=border_color, width=border_width)
    
    # Simulasi teks dengan masalah
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Teks dengan masalah (posisi tidak tepat, warna buram, dll)
    text_lines = [
        "PRODUK BERKUALITAS",
        "Made in Indonesia", 
        "Exp: 12/2025",
        "Net: 500g"
    ]
    
    y_pos = 30
    for i, line in enumerate(text_lines):
        # Posisi tidak tepat
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x_pos = (width - text_width) // 2 + random.randint(-10, 10)
        y_pos_offset = y_pos + random.randint(-5, 5)
        
        # Warna teks buram atau tidak konsisten
        text_color = (
            random.randint(50, 150),
            random.randint(50, 150), 
            random.randint(50, 150)
        )
        
        draw.text((x_pos, y_pos_offset), line, fill=text_color, font=font)
        y_pos += 35
    
    # Defect type 2: Noda atau goresan
    defect_types = random.choice(['noda', 'goresan', 'robek'])
    
    if defect_types == 'noda':
        # Tambahkan noda
        for _ in range(random.randint(2, 5)):
            noda_x = random.randint(20, width-40)
            noda_y = random.randint(20, height-40)
            noda_size = random.randint(5, 15)
            noda_color = (
                random.randint(100, 200),
                random.randint(100, 150),
                random.randint(50, 100)
            )
            draw.ellipse([noda_x, noda_y, noda_x + noda_size, noda_y + noda_size], 
                        fill=noda_color)
    
    elif defect_types == 'goresan':
        # Tambahkan goresan
        for _ in range(random.randint(1, 3)):
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            end_x = start_x + random.randint(-50, 50)
            end_y = start_y + random.randint(-50, 50)
            
            draw.line([start_x, start_y, end_x, end_y], 
                     fill=(150, 150, 150), width=random.randint(2, 5))
    
    elif defect_types == 'robek':
        # Simulasi robekan dengan area putih tidak beraturan
        robek_points = []
        center_x = random.randint(50, width-50)
        center_y = random.randint(50, height-50)
        
        for angle in range(0, 360, 30):
            radius = random.randint(10, 25)
            x = center_x + int(radius * np.cos(np.radians(angle)))
            y = center_y + int(radius * np.sin(np.radians(angle)))
            robek_points.append((x, y))
        
        draw.polygon(robek_points, fill=(240, 240, 240))
    
    # Tambahkan noise lebih banyak untuk defect
    noise_level = 15
    for _ in range(50):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        noise_color = (
            min(255, max(0, 255 + random.randint(-noise_level, noise_level))),
            min(255, max(0, 255 + random.randint(-noise_level, noise_level))),
            min(255, max(0, 255 + random.randint(-noise_level, noise_level)))
        )
        draw.point((x, y), fill=noise_color)
    
    return img

def create_sample_dataset(good_count=15, defect_count=15):
    """
    Membuat dataset sample untuk training
    
    Args:
        good_count (int): Jumlah gambar good yang akan dibuat
        defect_count (int): Jumlah gambar defect yang akan dibuat
    """
    print("=== Membuat Sample Dataset untuk Training ===")
    
    # Buat folder training data
    base_folder = "static/training_data"
    good_folder = os.path.join(base_folder, "good")
    defect_folder = os.path.join(base_folder, "defect")
    
    os.makedirs(good_folder, exist_ok=True)
    os.makedirs(defect_folder, exist_ok=True)
    
    print(f"\n1. Membuat {good_count} gambar GOOD...")
    
    # Buat gambar good
    for i in range(good_count):
        img = create_sample_good_image()
        filename = f"sample_good_{i+1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(good_folder, filename)
        img.save(filepath)
        print(f"   ✓ Dibuat: {filename}")
    
    print(f"\n2. Membuat {defect_count} gambar DEFECT...")
    
    # Buat gambar defect
    for i in range(defect_count):
        img = create_sample_defect_image()
        filename = f"sample_defect_{i+1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(defect_folder, filename)
        img.save(filepath)
        print(f"   ✓ Dibuat: {filename}")
    
    print(f"\n✅ Sample dataset berhasil dibuat!")
    print(f"📁 Lokasi: {os.path.abspath(base_folder)}")
    print(f"📊 Total: {good_count} good + {defect_count} defect = {good_count + defect_count} gambar")
    print(f"\n💡 Tips:")
    print(f"   - Akses halaman training di: http://localhost:5000/training")
    print(f"   - Upload gambar real untuk hasil yang lebih baik")
    print(f"   - Minimal 10 gambar per kategori untuk training optimal")

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
    # Buat sample dataset
    create_sample_dataset(good_count=15, defect_count=15)
