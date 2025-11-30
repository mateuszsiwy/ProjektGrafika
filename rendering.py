#!/usr/bin/env python3
"""
Funkcje rysowania sceny OpenGL
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from shader_utils import create_model_matrix


def draw_cube_edges(cube_size):
    """Narysuj krawędzie sześcianu"""
    half_size = cube_size / 2
    
    glDisable(GL_LIGHTING)
    glColor3f(1.0, 1.0, 1.0)
    glLineWidth(2.0)
    
    glBegin(GL_LINES)
    vertices = [
        [-half_size, -half_size, -half_size],
        [half_size, -half_size, -half_size],
        [half_size, half_size, -half_size],
        [-half_size, half_size, -half_size],
        [-half_size, -half_size, half_size],
        [half_size, -half_size, half_size],
        [half_size, half_size, half_size],
        [-half_size, half_size, half_size],
    ]
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    for edge in edges:
        glVertex3fv(vertices[edge[0]])
        glVertex3fv(vertices[edge[1]])
    
    glEnd()
    glLineWidth(1.0)
    glEnable(GL_LIGHTING)


def draw_shadow(ball_position, ball_radius, plane_normal, plane_point, cube_size):
    """Narysuj cień kulki na ścianie sześcianu"""
    half_size = cube_size / 2
    
    plane_normal = np.array(plane_normal)
    plane_point = np.array(plane_point)
    
    ball_center = np.array(ball_position)
    d = np.dot(plane_normal, plane_point)
    t = (d - np.dot(plane_normal, ball_center)) / np.dot(plane_normal, plane_normal)
    
    shadow_center = ball_center + t * plane_normal
    
    distance_to_plane = abs(t)
    
    if distance_to_plane > ball_radius * 5:
        return
    
    shadow_alpha = max(0.0, min(0.95, 1.0 - (distance_to_plane / (ball_radius * 5))))
    
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.0, 0.0, 0.0, shadow_alpha)
    
    u = np.array([1.0, 0.0, 0.0]) if abs(plane_normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = u - np.dot(u, plane_normal) * plane_normal
    u = u / np.linalg.norm(u)
    v = np.cross(plane_normal, u)
    
    glBegin(GL_TRIANGLE_FAN)
    glVertex3fv(shadow_center)
    
    num_segments = 32
    for i in range(num_segments + 1):
        angle = 2.0 * np.pi * i / num_segments
        offset = ball_radius * (np.cos(angle) * u + np.sin(angle) * v)
        point = shadow_center + offset
        
        clamped_point = np.clip(point, -half_size, half_size)
        glVertex3fv(clamped_point)
    
    glEnd()
    
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)


def render_with_shaders(balls, shader_program, sphere_vao, sphere_indices_count, 
                        projection_matrix, view_matrix, cube_rotation_axis, cube_rotation_angle,
                        textures, setup_shader_lights_func):
    """Renderuj kulki z użyciem shaderów"""
    glUseProgram(shader_program)
    
    proj_loc = glGetUniformLocation(shader_program, "projection")
    view_loc = glGetUniformLocation(shader_program, "view")
    model_loc = glGetUniformLocation(shader_program, "model")
    
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix)
    
    setup_shader_lights_func(shader_program, balls)
    
    glBindVertexArray(sphere_vao)
    
    for ball in balls:
        if ball.is_light:
            continue
        
        model_matrix = create_model_matrix(
            ball.position,
            ball.radius,
            cube_rotation_axis,
            cube_rotation_angle
        )
        
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_matrix)
        
        color_loc = glGetUniformLocation(shader_program, "objectColor")
        glUniform3f(color_loc, *ball.color)
        
        if ball.texture_id is not None:
            use_texture_loc = glGetUniformLocation(shader_program, "useTexture")
            glUniform1i(use_texture_loc, 1)
            
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, ball.texture_id)
            texture_loc = glGetUniformLocation(shader_program, "textureSampler")
            glUniform1i(texture_loc, 0)
        else:
            use_texture_loc = glGetUniformLocation(shader_program, "useTexture")
            glUniform1i(use_texture_loc, 0)
        
        glDrawElements(GL_TRIANGLES, sphere_indices_count, GL_UNSIGNED_INT, None)
    
    glBindVertexArray(0)
    glUseProgram(0)
