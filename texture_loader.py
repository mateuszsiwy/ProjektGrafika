#!/usr/bin/env python3
"""
Funkcje ładowania tekstur
"""

import os
from PIL import Image
from OpenGL.GL import *


def load_texture(filename):
    """Załaduj teksturę z pliku PNG"""
    if not os.path.exists(filename):
        print(f"Plik tekstury nie istnieje: {filename}")
        return None
    
    try:
        image = Image.open(filename)
        image = image.convert("RGB")
        img_data = image.tobytes("raw", "RGB", 0, -1)
        
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        
        return texture
    except Exception as e:
        print(f"Błąd ładowania tekstury {filename}: {e}")
        return None


def load_textures(texture_folder="textures"):
    """Załaduj wszystkie tekstury z folderu"""
    textures = []
    
    if not os.path.exists(texture_folder):
        print(f"Folder tekstur nie istnieje: {texture_folder}")
        return textures
    
    texture_files = sorted([f for f in os.listdir(texture_folder) if f.endswith('.png')])
    
    for texture_file in texture_files:
        texture_path = os.path.join(texture_folder, texture_file)
        texture = load_texture(texture_path)
        if texture:
            textures.append(texture)
    
    return textures
