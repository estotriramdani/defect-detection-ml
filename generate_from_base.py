"""
Script untuk generate training data berdasarkan gambar base label_fibemini.png
Membuat variasi good dan defect dari gambar base
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random
from datetime import datetime

# Pastikan output UTF-8 agar emoji tidak error di konsol Windows (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def load_base_image(base_path):
    """
    Load gambar base dan return PIL Image
    
    Args:
        base_path (str): Path ke gambar base
        
    Returns:
        PIL.Image: Gambar base
    """
    try:
        img = Image.open(base_path).convert('RGB')
        print(f"✅ Base image loaded: {img.size}")
        return img
    except Exception as e:
        print(f"❌ Error loading base image: {e}")
        return None

def create_good_variations(base_img, count=20):
    """
    Buat variasi gambar good dari base image
    
    Args:
        base_img (PIL.Image): Gambar base
        count (int): Jumlah variasi yang akan dibuat
        
    Returns:
        list: List PIL Images
    """
    variations = []
    
    # TAMBAHKAN GAMBAR BASE ASLI SEBAGAI GOOD PERTAMA
    variations.append(base_img.copy())
    print(f"  ✓ Original base image added as good variation 1/{count+1}")
    
    for i in range(count):
        img = base_img.copy()
        
        # Variasi 1: Slight brightness adjustment
        enhancer = ImageEnhance.Brightness(img)
        brightness_factor = random.uniform(0.95, 1.05)  # Lebih subtle
        img = enhancer.enhance(brightness_factor)
        
        # Variasi 2: Slight contrast adjustment
        enhancer = ImageEnhance.Contrast(img)
        contrast_factor = random.uniform(0.95, 1.05)  # Lebih subtle
        img = enhancer.enhance(contrast_factor)
        
        # Variasi 3: Slight color adjustment
        enhancer = ImageEnhance.Color(img)
        color_factor = random.uniform(0.98, 1.02)  # Lebih subtle
        img = enhancer.enhance(color_factor)
        
        # Variasi 4: Minor rotation (±1 degrees) - lebih kecil
        rotation_angle = random.uniform(-1, 1)
        img = img.rotate(rotation_angle, fillcolor='white')
        
        # Variasi 5: Slight resize and crop back (lebih subtle)
        if random.choice([True, False]):
            old_size = img.size
            scale_factor = random.uniform(1.01, 1.03)  # Lebih kecil
            new_size = (int(old_size[0] * scale_factor), int(old_size[1] * scale_factor))
            img = img.resize(new_size)
            
            # Crop back to original size
            left = (new_size[0] - old_size[0]) // 2
            top = (new_size[1] - old_size[1]) // 2
            img = img.crop((left, top, left + old_size[0], top + old_size[1]))
        
        # Variasi 6: Minor noise (very subtle)
        if random.choice([True, False]):
            img_array = np.array(img)
            noise = np.random.normal(0, 1, img_array.shape).astype(np.int16)  # Noise lebih kecil
            img_array = img_array.astype(np.int16) + noise
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        variations.append(img)
        print(f"  ✓ Good variation {i+2}/{count+1} created")
    
    return variations

def create_defect_variations(base_img, count=20):
    """
    Buat variasi gambar defect dari base image
    
    Args:
        base_img (PIL.Image): Gambar base
        count (int): Jumlah variasi yang akan dibuat
        
    Returns:
        list: List PIL Images
    """
    variations = []
    
    for i in range(count):
        img = base_img.copy()
        draw = ImageDraw.Draw(img)
        
        # Pilih jenis defect secara random
        defect_types = ['stain', 'scratch', 'tear', 'fade', 'blur', 'misprint', 'wrinkle']
        selected_defects = random.sample(defect_types, random.randint(1, 3))
        
        for defect_type in selected_defects:
            
            if defect_type == 'stain':
                # Tambahkan noda
                for _ in range(random.randint(1, 4)):
                    x = random.randint(0, img.width - 30)
                    y = random.randint(0, img.height - 30)
                    size = random.randint(8, 25)
                    
                    # Warna noda (coklat, abu-abu, kuning)
                    stain_colors = [
                        (139, 109, 83),   # Brown
                        (105, 105, 105),  # Gray
                        (184, 134, 11),   # Dark yellow
                        (120, 81, 45),    # Dark brown
                    ]
                    color = random.choice(stain_colors)
                    
                    # Gambar noda dengan bentuk tidak beraturan
                    points = []
                    for angle in range(0, 360, 30):
                        radius = random.randint(size//2, size)
                        px = x + int(radius * np.cos(np.radians(angle)))
                        py = y + int(radius * np.sin(np.radians(angle)))
                        points.append((px, py))
                    
                    draw.polygon(points, fill=color)
            
            elif defect_type == 'scratch':
                # Tambahkan goresan
                for _ in range(random.randint(1, 3)):
                    x1 = random.randint(0, img.width)
                    y1 = random.randint(0, img.height)
                    
                    # Goresan dengan arah random tapi tidak terlalu panjang
                    length = random.randint(20, 80)
                    angle = random.randint(0, 360)
                    x2 = x1 + int(length * np.cos(np.radians(angle)))
                    y2 = y1 + int(length * np.sin(np.radians(angle)))
                    
                    # Warna goresan
                    scratch_color = (random.randint(100, 150), random.randint(100, 150), random.randint(100, 150))
                    width = random.randint(2, 5)
                    
                    draw.line([x1, y1, x2, y2], fill=scratch_color, width=width)
            
            elif defect_type == 'tear':
                # Simulasi robekan
                tear_x = random.randint(20, img.width - 40)
                tear_y = random.randint(20, img.height - 40)
                
                # Bentuk robekan tidak beraturan
                tear_points = []
                for j in range(8):
                    angle = j * 45
                    radius = random.randint(10, 25)
                    px = tear_x + int(radius * np.cos(np.radians(angle)))
                    py = tear_y + int(radius * np.sin(np.radians(angle)))
                    tear_points.append((px, py))
                
                draw.polygon(tear_points, fill=(240, 240, 240))
            
            elif defect_type == 'fade':
                # Simulasi warna pudar
                img_array = np.array(img)
                
                # Area yang akan fade
                fade_x = random.randint(0, img.width//2)
                fade_y = random.randint(0, img.height//2)
                fade_w = random.randint(40, 100)
                fade_h = random.randint(40, 100)
                
                # Apply fading effect
                fade_factor = random.uniform(1.3, 1.8)
                img_array[fade_y:fade_y+fade_h, fade_x:fade_x+fade_w] = np.clip(
                    img_array[fade_y:fade_y+fade_h, fade_x:fade_x+fade_w] * fade_factor, 0, 255
                )
                
                img = Image.fromarray(img_array.astype(np.uint8))
                draw = ImageDraw.Draw(img)
            
            elif defect_type == 'blur':
                # Area blur
                blur_x = random.randint(0, img.width//2)
                blur_y = random.randint(0, img.height//2)
                blur_w = random.randint(30, 80)
                blur_h = random.randint(20, 60)
                
                # Extract area dan blur
                area = img.crop((blur_x, blur_y, blur_x + blur_w, blur_y + blur_h))
                blurred_area = area.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 4)))
                img.paste(blurred_area, (blur_x, blur_y))
                draw = ImageDraw.Draw(img)
            
            elif defect_type == 'misprint':
                # Simulasi misprint dengan duplikasi teks atau gambar
                misprint_color = (random.randint(50, 100), random.randint(50, 100), random.randint(50, 100))
                
                # Random shapes untuk misprint
                for _ in range(random.randint(2, 5)):
                    x = random.randint(0, img.width - 20)
                    y = random.randint(0, img.height - 20)
                    w = random.randint(10, 30)
                    h = random.randint(5, 15)
                    
                    draw.rectangle([x, y, x+w, y+h], fill=misprint_color)
            
            elif defect_type == 'wrinkle':
                # Simulasi kerutan dengan garis-garis
                for _ in range(random.randint(3, 7)):
                    # Garis kerutan
                    x1 = random.randint(0, img.width)
                    y1 = random.randint(0, img.height)
                    
                    # Kerutan biasanya mengikuti pola tertentu
                    direction = random.choice(['horizontal', 'vertical', 'diagonal'])
                    
                    if direction == 'horizontal':
                        x2 = x1 + random.randint(30, 100)
                        y2 = y1 + random.randint(-10, 10)
                    elif direction == 'vertical':
                        x2 = x1 + random.randint(-10, 10)
                        y2 = y1 + random.randint(30, 100)
                    else:  # diagonal
                        x2 = x1 + random.randint(30, 80)
                        y2 = y1 + random.randint(30, 80)
                    
                    wrinkle_color = (random.randint(180, 220), random.randint(180, 220), random.randint(180, 220))
                    draw.line([x1, y1, x2, y2], fill=wrinkle_color, width=random.randint(1, 3))
        
        variations.append(img)
        print(f"  ✓ Defect variation {i+1}/{count} created ({', '.join(selected_defects)})")
    
    return variations

def save_variations(variations, folder_path, prefix, category):
    """
    Simpan variasi gambar ke folder
    
    Args:
        variations (list): List PIL Images
        folder_path (str): Path folder tujuan
        prefix (str): Prefix nama file
        category (str): Kategori (good/defect)
    """
    os.makedirs(folder_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, img in enumerate(variations):
        filename = f"{prefix}_{category}_{i+1:03d}_{timestamp}.png"
        filepath = os.path.join(folder_path, filename)
        img.save(filepath)
        print(f"  💾 Saved: {filename}")

def generate_training_data_from_base(base_image_path, good_count=20, defect_count=20):
    """
    Generate training data dari base image
    
    Args:
        base_image_path (str): Path ke gambar base
        good_count (int): Jumlah variasi good
        defect_count (int): Jumlah variasi defect
    """
    print("🚀 Starting training data generation from base image...")
    print(f"📁 Base image: {base_image_path}")
    
    # Load base image
    base_img = load_base_image(base_image_path)
    if base_img is None:
        return False
    
    # Generate good variations
    print(f"\n📸 Generating {good_count} GOOD variations...")
    good_variations = create_good_variations(base_img, good_count)
    
    # Generate defect variations
    print(f"\n⚠️  Generating {defect_count} DEFECT variations...")
    defect_variations = create_defect_variations(base_img, defect_count)
    
    # Save variations
    print(f"\n💾 Saving variations...")
    
    good_folder = "static/training_data/good"
    defect_folder = "static/training_data/defect"
    
    save_variations(good_variations, good_folder, "fibemini", "good")
    save_variations(defect_variations, defect_folder, "fibemini", "defect")
    
    print(f"\n✅ Training data generation completed!")
    print(f"📊 Summary:")
    print(f"   - Good samples: {good_count}")
    print(f"   - Defect samples: {defect_count}")
    print(f"   - Total: {good_count + defect_count} images")
    print(f"📁 Locations:")
    print(f"   - Good: {os.path.abspath(good_folder)}")
    print(f"   - Defect: {os.path.abspath(defect_folder)}")
    
    return True

if __name__ == "__main__":
    # Path ke base image
    base_image_path = "static/base-data/label_fibemini.png"
    
    # Generate training data
    success = generate_training_data_from_base(
        base_image_path=base_image_path,
        good_count=25,  # 25 good samples
        defect_count=25  # 25 defect samples
    )
    
    if success:
        print(f"\n🎉 Ready for training!")
        print(f"💡 Next steps:")
        print(f"   1. Run: python retrain_model.py")
        print(f"   2. Or use web interface: http://localhost:5001/training")
    else:
        print(f"\n❌ Failed to generate training data")
