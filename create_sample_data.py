"""
Script untuk membuat sample data training
Menghasilkan gambar simulasi untuk kategori 'good' dan 'defect'
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from datetime import datetime

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

if __name__ == "__main__":
    # Buat sample dataset
    create_sample_dataset(good_count=15, defect_count=15)
