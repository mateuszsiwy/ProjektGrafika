#!/usr/bin/env python3
"""
Klasa Ball - reprezentuje kulkę z fizyką
"""

import numpy as np
import random
from OpenGL.GL import *
from OpenGL.GLU import *


class Ball:
    """Klasa reprezentująca kulkę z fizyką"""
    
    def __init__(self, position, velocity, radius=0.3, color=None, texture_id=None, is_light=False):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = radius
        self.mass = radius ** 3
        self.texture_id = texture_id
        self.is_light = is_light
        
        if color is None:
            self.color = [random.random(), random.random(), random.random()]
        else:
            self.color = color
    
    def update(self, dt, gravity, cube_size, cube_rotation_speed, cube_rotation_axis, cube_rotation_angle):
        """Aktualizuj pozycję kulki"""
        angle_rad = np.radians(-cube_rotation_angle)
        axis = np.array(cube_rotation_axis)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        rotation_matrix = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
        local_gravity = np.dot(rotation_matrix, gravity)
        
        self.velocity += local_gravity * dt
        
        half_size = cube_size / 2
        touching_wall = False
        
        for i in range(3):
            if abs(self.position[i]) + self.radius >= half_size - 0.01:
                touching_wall = True
                break
        
        if touching_wall and abs(cube_rotation_speed) > 0.01:
            omega_deg = cube_rotation_speed
            omega = np.radians(omega_deg)
            omega_vec = np.array(cube_rotation_axis) * omega
            tangential_velocity = np.cross(omega_vec, self.position)
            friction_coefficient = 0.3
            self.velocity += (tangential_velocity - self.velocity) * friction_coefficient * dt
        
        self.position += self.velocity * dt
    
    def check_cube_collision(self, cube_size):
        """Sprawdź kolizję ze ścianami sześcianu"""
        half_size = cube_size / 2
        damping = 0.8
        
        for i in range(3):
            if self.position[i] - self.radius < -half_size:
                self.position[i] = -half_size + self.radius
                self.velocity[i] = abs(self.velocity[i]) * damping
            elif self.position[i] + self.radius > half_size:
                self.position[i] = half_size - self.radius
                self.velocity[i] = -abs(self.velocity[i]) * damping
    
    def check_ball_collision(self, other):
        """Sprawdź i rozwiąż kolizję z inną kulką"""
        diff = self.position - other.position
        distance = np.linalg.norm(diff)
        min_distance = self.radius + other.radius
        
        if distance < min_distance and distance > 0:
            normal = diff / distance
            overlap = min_distance - distance
            self.position += normal * (overlap / 2)
            other.position -= normal * (overlap / 2)
            
            relative_velocity = self.velocity - other.velocity
            velocity_along_normal = np.dot(relative_velocity, normal)
            
            if velocity_along_normal > 0:
                return
            
            restitution = 0.9
            impulse = (-(1 + restitution) * velocity_along_normal) / (1/self.mass + 1/other.mass)
            
            self.velocity += (impulse / self.mass) * normal
            other.velocity -= (impulse / other.mass) * normal
    
    def draw(self):
        """Narysuj kulkę"""
        glPushMatrix()
        glTranslatef(*self.position)
        
        if self.is_light:
            glDisable(GL_LIGHTING)
            glColor3f(*self.color)
            
            quadric = gluNewQuadric()
            gluSphere(quadric, self.radius, 32, 32)
            gluDeleteQuadric(quadric)
            
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(self.color[0], self.color[1], self.color[2], 0.3)
            quadric2 = gluNewQuadric()
            gluSphere(quadric2, self.radius * 1.8, 32, 32)
            gluDeleteQuadric(quadric2)
            
            glEnable(GL_LIGHTING)
        else:
            material_ambient = self.color + [1.0]
            material_diffuse = self.color + [1.0]
            material_specular = [1.0, 1.0, 1.0, 1.0]
            material_shininess = [32.0]
            
            glMaterialfv(GL_FRONT, GL_AMBIENT, material_ambient)
            glMaterialfv(GL_FRONT, GL_DIFFUSE, material_diffuse)
            glMaterialfv(GL_FRONT, GL_SPECULAR, material_specular)
            glMaterialfv(GL_FRONT, GL_SHININESS, material_shininess)
            
            if self.texture_id:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, self.texture_id)
            
            glColor3f(*self.color)
            
            quadric = gluNewQuadric()
            gluQuadricTexture(quadric, GL_TRUE)
            gluQuadricNormals(quadric, GLU_SMOOTH)
            gluSphere(quadric, self.radius, 32, 32)
            gluDeleteQuadric(quadric)
            
            if self.texture_id:
                glDisable(GL_TEXTURE_2D)
        
        glPopMatrix()
