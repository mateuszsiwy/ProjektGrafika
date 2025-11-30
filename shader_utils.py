#!/usr/bin/env python3
"""
Funkcje pomocnicze do zarządzania shaderami i geometrią
"""

import numpy as np
from OpenGL.GL import *


def load_shader(shader_file, shader_type):
    """Załaduj i skompiluj shader"""
    try:
        with open(shader_file, 'r') as f:
            shader_source = f.read()
    except FileNotFoundError:
        print(f"Nie znaleziono pliku shadera: {shader_file}")
        return None
    
    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_source)
    glCompileShader(shader)
    
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        print(f"Błąd kompilacji shadera {shader_file}:\n{error}")
        return None
    
    return shader


def create_shader_program(vertex_shader_path, fragment_shader_path):
    """Utwórz program shaderów"""
    vertex_shader = load_shader(vertex_shader_path, GL_VERTEX_SHADER)
    fragment_shader = load_shader(fragment_shader_path, GL_FRAGMENT_SHADER)
    
    if not vertex_shader or not fragment_shader:
        return None
    
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        print(f"Błąd linkowania programu:\n{error}")
        return None
    
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    
    return program


def create_sphere_geometry(radius=1.0, slices=32, stacks=32):
    """Generuj geometrię sfery dla VAO/VBO"""
    vertices = []
    normals = []
    texcoords = []
    indices = []
    
    for i in range(stacks + 1):
        phi = np.pi * i / stacks
        for j in range(slices + 1):
            theta = 2.0 * np.pi * j / slices
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            
            vertices.extend([x, y, z])
            normals.extend([x/radius, y/radius, z/radius])
            texcoords.extend([j / slices, i / stacks])
    
    for i in range(stacks):
        for j in range(slices):
            first = i * (slices + 1) + j
            second = first + slices + 1
            
            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])
    
    return (np.array(vertices, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(texcoords, dtype=np.float32),
            np.array(indices, dtype=np.uint32))


def create_vao_vbo(vertices, normals, texcoords, indices):
    """Utwórz VAO i VBO dla geometrii"""
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    
    vbo_vertices = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    
    vbo_normals = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)
    
    vbo_texcoords = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoords)
    glBufferData(GL_ARRAY_BUFFER, texcoords.nbytes, texcoords, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(2)
    
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    
    glBindVertexArray(0)
    
    return vao, len(indices)


def create_model_matrix(position, scale, cube_rotation_axis, cube_rotation_angle):
    """Utwórz macierz modelu z rotacją Rodriguesa"""
    angle_rad = np.radians(cube_rotation_angle)
    axis = np.array(cube_rotation_axis)
    
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    rotation_3x3 = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
    
    model_matrix = np.eye(4, dtype=np.float32)
    model_matrix[:3, :3] = rotation_3x3
    model_matrix[:3, 3] = position
    
    scale_matrix = np.eye(4, dtype=np.float32)
    scale_matrix[0, 0] = scale
    scale_matrix[1, 1] = scale
    scale_matrix[2, 2] = scale
    
    return np.dot(model_matrix, scale_matrix)
