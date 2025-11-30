#!/usr/bin/env python3
import sys
import numpy as np
import random
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

from ball import Ball
from shader_utils import create_shader_program, create_sphere_geometry, create_vao_vbo, create_model_matrix
from rendering import draw_cube_edges, draw_shadow, render_with_shaders
from texture_loader import load_textures


class GLWidget(QOpenGLWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.gravity = np.array([0.0, -9.81, 0.0])
        self.cube_size = 10.0
        
        self.cube_rotation_angle = 0.0
        self.cube_rotation_speed = 0.0
        self.cube_rotation_axis = [1.0, 0.0, 0.0]
        
        self.camera_distance = 20.0
        self.camera_rotation_x = 20.0
        self.camera_rotation_y = 45.0
        
        self.max_lights = 8
        
        self.shader_program = None
        self.use_shaders = True
        self.shadow_enabled = True
        
        self.sphere_vao = None
        self.sphere_vbo_vertices = None
        self.sphere_vbo_normals = None
        self.sphere_vbo_texcoords = None
        self.sphere_ebo = None
        self.sphere_index_count = 0
        
        self.textures = []
        
        self.balls = []
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(16)
        
        self.last_time = 0
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()
    
    def init_balls(self, count):
        self.balls = []
        half_size = self.cube_size / 2 - 1.0
        
        light_colors = [
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.5, 1.0],
        ]
        
        for i, light_color in enumerate(light_colors):
            position = [
                random.uniform(-half_size, half_size),
                random.uniform(-half_size, half_size),
                random.uniform(-half_size, half_size)
            ]
            velocity = [
                random.uniform(-1.5, 1.5),
                random.uniform(-1.5, 1.5),
                random.uniform(-1.5, 1.5)
            ]
            self.balls.append(Ball(position, velocity, radius=0.4, color=light_color, is_light=True))
        
        for i in range(count):
            position = [
                random.uniform(-half_size, half_size),
                random.uniform(-half_size, half_size),
                random.uniform(-half_size, half_size)
            ]
            velocity = [
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                random.uniform(-2, 2)
            ]
            radius = random.uniform(0.2, 0.4)
            texture_id = self.textures[i % len(self.textures)] if self.textures else None
            
            self.balls.append(Ball(position, velocity, radius, texture_id=texture_id))
    
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        
        try:
            self.shader_program = create_shader_program(
                'shaders/vertex_shader.glsl',
                'shaders/fragment_shader.glsl'
            )
            if self.shader_program:
                self.use_shaders = True
                self.setup_sphere_geometry()
            else:
                self.use_shaders = False
                self.setup_classic_lighting()
        except Exception as e:
            self.use_shaders = False
            self.setup_classic_lighting()
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glClearColor(0.05, 0.05, 0.1, 1.0)
        
        self.textures = load_textures()
        
        self.init_balls(12)
        
    def setup_sphere_geometry(self):
        vertices, normals, texcoords, indices = create_sphere_geometry(1.0, 32, 32)
        self.sphere_vao, self.sphere_index_count = create_vao_vbo(vertices, normals, texcoords, indices)
        
        glBindVertexArray(0)
    
    def setup_classic_lighting(self):
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
    
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h != 0 else 1, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if self.use_shaders and self.shader_program:
            self.render_with_shaders()
        else:
            self.render_classic()
        
        self.update()
    
    def render_classic(self):
        glLoadIdentity()
        
        glTranslatef(0, 0, -self.camera_distance)
        glRotatef(self.camera_rotation_x, 1, 0, 0)
        glRotatef(self.camera_rotation_y, 0, 1, 0)
        
        self.setup_dynamic_lights()
        
        self.draw_cube()
        
        if self.shadow_enabled:
            self.draw_all_shadows()
        
        glPushMatrix()
        glRotatef(self.cube_rotation_angle, *self.cube_rotation_axis)
        
        for ball in self.balls:
            ball.draw()
        
        glPopMatrix()
    
    def render_with_shaders(self):
        """Renderowanie z użyciem shaderów GLSL"""
        glUseProgram(self.shader_program)
        
        glLoadIdentity()
        glTranslatef(0, 0, -self.camera_distance)
        glRotatef(self.camera_rotation_x, 1, 0, 0)
        glRotatef(self.camera_rotation_y, 0, 1, 0)
        
        view_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)
        projection_matrix = glGetFloatv(GL_PROJECTION_MATRIX)
        
        view_loc = glGetUniformLocation(self.shader_program, "view")
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        if view_loc != -1:
            glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix)
        if proj_loc != -1:
            glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix)
        
        camera_pos_loc = glGetUniformLocation(self.shader_program, "viewPos")
        if camera_pos_loc != -1:
            glUniform3f(camera_pos_loc, 0.0, 0.0, self.camera_distance)
        
        self.setup_shader_lights()
        
        glUseProgram(0)
        self.draw_cube()
        
        if self.shadow_enabled:
            self.draw_all_shadows()
        
        glUseProgram(self.shader_program)
        
        for ball in self.balls:
            if ball.is_light:
                glUseProgram(0)
                glPushMatrix()
                glRotatef(self.cube_rotation_angle, *self.cube_rotation_axis)
                ball.draw()
                glPopMatrix()
                glUseProgram(self.shader_program)
            else:
                self.draw_ball_with_shader(ball)
        
        glUseProgram(0)
    
    def draw_ball_with_shader(self, ball):        
        angle_rad = np.radians(self.cube_rotation_angle)
        axis = np.array(self.cube_rotation_axis, dtype=np.float32)
        axis = axis / np.linalg.norm(axis)
        
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        x, y, z = axis
        
        cube_rotation = np.array([
            [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s, 0],
            [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s, 0],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        translation = np.eye(4, dtype=np.float32)
        translation[:3, 3] = ball.position
        
        scale = np.eye(4, dtype=np.float32)
        scale[0, 0] = ball.radius
        scale[1, 1] = ball.radius
        scale[2, 2] = ball.radius
        
        model_matrix = cube_rotation @ translation @ scale
        
        model_loc = glGetUniformLocation(self.shader_program, "model")
        if model_loc != -1:
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_matrix.T)
        
        normal_matrix_4x4 = cube_rotation @ translation
        normal_matrix = normal_matrix_4x4[:3, :3]
        normal_matrix = np.linalg.inv(normal_matrix).T
        
        normal_loc = glGetUniformLocation(self.shader_program, "normalMatrix")
        if normal_loc != -1:
            glUniformMatrix3fv(normal_loc, 1, GL_FALSE, normal_matrix)
        
        color_loc = glGetUniformLocation(self.shader_program, "materialColor")
        if color_loc != -1:
            glUniform3f(color_loc, *ball.color)
        
        shininess_loc = glGetUniformLocation(self.shader_program, "shininess")
        if shininess_loc != -1:
            glUniform1f(shininess_loc, 32.0)
        
        use_texture_loc = glGetUniformLocation(self.shader_program, "useTexture")
        if use_texture_loc != -1:
            if ball.texture_id:
                glUniform1i(use_texture_loc, 1)
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, ball.texture_id)
                tex_loc = glGetUniformLocation(self.shader_program, "texture1")
                if tex_loc != -1:
                    glUniform1i(tex_loc, 0)
            else:
                glUniform1i(use_texture_loc, 0)
        
        glBindVertexArray(self.sphere_vao)
        glDrawElements(GL_TRIANGLES, self.sphere_index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
    
    def setup_shader_lights(self):
        light_positions = []
        light_colors = []
        
        for ball in self.balls:
            if ball.is_light and len(light_positions) < self.max_lights:
                angle_rad = np.radians(self.cube_rotation_angle)
                axis = np.array(self.cube_rotation_axis)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                rotation_matrix = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
                global_pos = np.dot(rotation_matrix, ball.position)
                
                light_positions.append(global_pos)
                light_colors.append(np.array(ball.color) * 1.5)
        
        num_lights = len(light_positions)
        num_lights_loc = glGetUniformLocation(self.shader_program, "numLights")
        if num_lights_loc != -1:
            glUniform1i(num_lights_loc, num_lights)
        
        for i, pos in enumerate(light_positions):
            loc_name = f"lightPositions[{i}]"
            loc = glGetUniformLocation(self.shader_program, loc_name)
            if loc != -1:
                glUniform3f(loc, pos[0], pos[1], pos[2])
        
        for i, color in enumerate(light_colors):
            loc_name = f"lightColors[{i}]"
            loc = glGetUniformLocation(self.shader_program, loc_name)
            if loc != -1:
                glUniform3f(loc, color[0], color[1], color[2])
    
    def setup_dynamic_lights(self):
        """Ustaw światła od świecących kulek"""
        if not self.use_shaders:
            for i in range(self.max_lights):
                glDisable(GL_LIGHT0 + i)
            
            light_index = 0
            for ball in self.balls:
                if ball.is_light and light_index < self.max_lights:
                    angle_rad = np.radians(self.cube_rotation_angle)
                    axis = np.array(self.cube_rotation_axis)
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)
                    K = np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ])
                    rotation_matrix = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
                    global_pos = np.dot(rotation_matrix, ball.position)
                    
                    light_id = GL_LIGHT0 + light_index
                    glEnable(light_id)
                    
                    glLightfv(light_id, GL_POSITION, list(global_pos) + [1.0])
                    light_color = [c * 1.5 for c in ball.color] + [1.0]
                    glLightfv(light_id, GL_AMBIENT, [c * 0.3 for c in ball.color] + [1.0])
                    glLightfv(light_id, GL_DIFFUSE, light_color)
                    glLightfv(light_id, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
                    
                    glLightf(light_id, GL_CONSTANT_ATTENUATION, 1.0)
                    glLightf(light_id, GL_LINEAR_ATTENUATION, 0.05)
                    glLightf(light_id, GL_QUADRATIC_ATTENUATION, 0.01)
                    
                    light_index += 1
    
    def draw_all_shadows(self):
        for light_ball in self.balls:
            if light_ball.is_light:
                angle_rad = np.radians(self.cube_rotation_angle)
                axis = np.array(self.cube_rotation_axis)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                rotation_matrix = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
                global_light = np.dot(rotation_matrix, light_ball.position)
                
                self.draw_shadows(global_light[0], global_light[1], global_light[2])
    
    def draw_shadows(self, light_x, light_y, light_z):
        glPushMatrix()
        glRotatef(self.cube_rotation_angle, *self.cube_rotation_axis)
        
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        half_size = self.cube_size / 2
        
        angle_rad = np.radians(-self.cube_rotation_angle)
        axis = np.array(self.cube_rotation_axis)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
        light_local = np.dot(rotation_matrix, np.array([light_x, light_y, light_z]))
        
        for ball in self.balls:
            if ball.is_light:
                continue
            
            if light_local[1] > ball.position[1]:
                t = (ball.position[1] + half_size) / (ball.position[1] - light_local[1])
                if t > 0 and t < 10:
                    shadow_x = ball.position[0] + t * (light_local[0] - ball.position[0])
                    shadow_z = ball.position[2] + t * (light_local[2] - ball.position[2])
                    
                    if abs(shadow_x) < half_size + 2 and abs(shadow_z) < half_size + 2:
                        self.draw_shadow_blob(shadow_x, -half_size + 0.02, shadow_z, ball.radius)
            
            if light_local[1] < ball.position[1]:
                t = (ball.position[1] - half_size) / (ball.position[1] - light_local[1])
                if t > 0 and t < 10:
                    shadow_x = ball.position[0] + t * (light_local[0] - ball.position[0])
                    shadow_z = ball.position[2] + t * (light_local[2] - ball.position[2])
                    
                    if abs(shadow_x) < half_size + 2 and abs(shadow_z) < half_size + 2:
                        self.draw_shadow_blob(shadow_x, half_size - 0.02, shadow_z, ball.radius)
            
            if light_local[0] > ball.position[0]:
                t = (ball.position[0] + half_size) / (ball.position[0] - light_local[0])
                if t > 0 and t < 10:
                    shadow_y = ball.position[1] + t * (light_local[1] - ball.position[1])
                    shadow_z = ball.position[2] + t * (light_local[2] - ball.position[2])
                    
                    if abs(shadow_y) < half_size + 2 and abs(shadow_z) < half_size + 2:
                        glPushMatrix()
                        glTranslatef(-half_size + 0.02, shadow_y, shadow_z)
                        glRotatef(90, 0, 1, 0)
                        glColor4f(0.0, 0.0, 0.0, 0.9)
                        quadric = gluNewQuadric()
                        gluDisk(quadric, 0, ball.radius * 1.5, 32, 1)
                        gluDeleteQuadric(quadric)
                        glPopMatrix()
            
            if light_local[0] < ball.position[0]:
                t = (ball.position[0] - half_size) / (ball.position[0] - light_local[0])
                if t > 0 and t < 10:
                    shadow_y = ball.position[1] + t * (light_local[1] - ball.position[1])
                    shadow_z = ball.position[2] + t * (light_local[2] - ball.position[2])
                    
                    if abs(shadow_y) < half_size + 2 and abs(shadow_z) < half_size + 2:
                        glPushMatrix()
                        glTranslatef(half_size - 0.02, shadow_y, shadow_z)
                        glRotatef(-90, 0, 1, 0)
                        glColor4f(0.0, 0.0, 0.0, 0.9)
                        quadric = gluNewQuadric()
                        gluDisk(quadric, 0, ball.radius * 1.5, 32, 1)
                        gluDeleteQuadric(quadric)
                        glPopMatrix()
            
            if light_local[2] < ball.position[2]:
                t = (ball.position[2] - half_size) / (ball.position[2] - light_local[2])
                if t > 0 and t < 10:
                    shadow_x = ball.position[0] + t * (light_local[0] - ball.position[0])
                    shadow_y = ball.position[1] + t * (light_local[1] - ball.position[1])
                    
                    if abs(shadow_x) < half_size + 2 and abs(shadow_y) < half_size + 2:
                        glPushMatrix()
                        glTranslatef(shadow_x, shadow_y, half_size - 0.02)
                        glColor4f(0.0, 0.0, 0.0, 0.9)
                        quadric = gluNewQuadric()
                        gluDisk(quadric, 0, ball.radius * 1.5, 32, 1)
                        gluDeleteQuadric(quadric)
                        glPopMatrix()
            
            if light_local[2] > ball.position[2]:
                t = (ball.position[2] + half_size) / (ball.position[2] - light_local[2])
                if t > 0 and t < 10:
                    shadow_x = ball.position[0] + t * (light_local[0] - ball.position[0])
                    shadow_y = ball.position[1] + t * (light_local[1] - ball.position[1])
                    
                    if abs(shadow_x) < half_size + 2 and abs(shadow_y) < half_size + 2:
                        glPushMatrix()
                        glTranslatef(shadow_x, shadow_y, -half_size + 0.02)
                        glRotatef(180, 0, 1, 0)
                        glColor4f(0.0, 0.0, 0.0, 0.9)
                        quadric = gluNewQuadric()
                        gluDisk(quadric, 0, ball.radius * 1.5, 32, 1)
                        gluDeleteQuadric(quadric)
                        glPopMatrix()
        
        glDepthMask(GL_TRUE)
        glEnable(GL_LIGHTING)
        glPopMatrix()
    
    def draw_shadow_blob(self, x, y, z, radius):
        """Rysuj okrągły cień (blob) na płaszczyźnie poziomej"""
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(90, 1, 0, 0)
        
        glColor4f(0.0, 0.0, 0.0, 0.95) 
        quadric = gluNewQuadric()
        gluDisk(quadric, 0, radius * 1.5, 32, 1)
        gluDeleteQuadric(quadric)
        
        glColor4f(0.0, 0.0, 0.0, 0.5)
        quadric2 = gluNewQuadric()
        gluDisk(quadric2, radius * 1.5, radius * 2.0, 32, 1)
        gluDeleteQuadric(quadric2)
        
        glPopMatrix()
    
    def draw_cube(self):
        half_size = self.cube_size / 2
        
        glPushMatrix()
        glRotatef(self.cube_rotation_angle, *self.cube_rotation_axis)
        
        glDisable(GL_LIGHTING)
        
        glDepthMask(GL_FALSE)
        
        glLineWidth(2.0)
        glColor4f(0.3, 0.6, 0.9, 0.6)
        
        glBegin(GL_LINES)
        
        # Dolna podstawa
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, -half_size)
        
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, half_size)
        
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(-half_size, -half_size, half_size)
        
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(-half_size, -half_size, -half_size)
        
        # Górna podstawa
        glVertex3f(-half_size, half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        
        glVertex3f(half_size, half_size, -half_size)
        glVertex3f(half_size, half_size, half_size)
        
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, -half_size)
        
        # Pionowe krawędzie
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(-half_size, half_size, -half_size)
        
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        
        glEnd()
        
        glColor4f(0.2, 0.3, 0.5, 0.1)
        
        glBegin(GL_QUADS)
        
        # Przednia ściana
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        
        # Tylna ściana
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(-half_size, half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        glVertex3f(half_size, -half_size, -half_size)
        
        # Górna ściana
        glVertex3f(-half_size, half_size, -half_size)
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(half_size, half_size, -half_size)
        
        # Dolna ściana
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(-half_size, -half_size, half_size)
        
        # Prawa ściana
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(half_size, -half_size, half_size)
        
        # Lewa ściana
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, -half_size)
        
        glEnd()
        
        glDepthMask(GL_TRUE)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
    
    def update_simulation(self):
        """Aktualizuj symulację fizyki"""
        dt = 0.016 
        
        self.cube_rotation_angle += self.cube_rotation_speed * dt
        self.cube_rotation_angle %= 360
        
        for ball in self.balls:
            ball.update(dt, self.gravity, self.cube_size, 
                       self.cube_rotation_speed, self.cube_rotation_axis, 
                       self.cube_rotation_angle)
        
        for ball in self.balls:
            ball.check_cube_collision(self.cube_size)
        
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                self.balls[i].check_ball_collision(self.balls[j])
        
        self.update()
    
    def keyPressEvent(self, event):
        """Obsługa klawiatury"""
        key = event.key()
        
        if key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            half_size = self.cube_size / 2 - 1.0
            position = [
                random.uniform(-half_size, half_size),
                random.uniform(-half_size, half_size),
                random.uniform(-half_size, half_size)
            ]
            velocity = [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2)]
            radius = random.uniform(0.2, 0.4)
            texture_id = self.textures[len(self.balls) % len(self.textures)] if self.textures else None
            self.balls.append(Ball(position, velocity, radius, texture_id=texture_id))
            print(f"Liczba kulek: {len(self.balls)}")
        
        elif key == Qt.Key.Key_Minus and len(self.balls) > 0:
            self.balls.pop()
            print(f"Liczba kulek: {len(self.balls)}")
        
        elif key == Qt.Key.Key_Up:
            self.cube_rotation_speed += 10
            print(f"Prędkość rotacji: {self.cube_rotation_speed:.1f}°/s")
        
        elif key == Qt.Key.Key_Down:
            self.cube_rotation_speed -= 10
            print(f"Prędkość rotacji: {self.cube_rotation_speed:.1f}°/s")
        
        elif key == Qt.Key.Key_Space:
            self.cube_rotation_speed = 0
            print("Rotacja zatrzymana")
        
        elif key == Qt.Key.Key_L:
            half_size = self.cube_size / 2 - 1.0
            position = [random.uniform(-half_size, half_size) for _ in range(3)]
            velocity = [random.uniform(-1.5, 1.5) for _ in range(3)]
            color = [random.random(), random.random(), random.random()]
            self.balls.append(Ball(position, velocity, radius=0.4, color=color, is_light=True))
            print(f"Dodano świecącą kulkę. Świecących: {sum(1 for b in self.balls if b.is_light)}")
        
        elif key == Qt.Key.Key_S:
            self.shadow_enabled = not self.shadow_enabled
        
        elif key == Qt.Key.Key_X:
            self.cube_rotation_axis = [1.0, 0.0, 0.0]
            print("Oś rotacji: X")
        
        elif key == Qt.Key.Key_Y:
            self.cube_rotation_axis = [0.0, 1.0, 0.0]
            print("Oś rotacji: Y")
        
        elif key == Qt.Key.Key_Z:
            self.cube_rotation_axis = [0.0, 0.0, 1.0]
            print("Oś rotacji: Z")
        
        # Reset
        elif key == Qt.Key.Key_R:
            self.cube_rotation_angle = 0
            self.cube_rotation_speed = 0
            self.init_balls(15)
            print("Reset symulacji")
        
        self.update()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.last_mouse_pos:
            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            
            self.camera_rotation_y += dx * 0.5
            self.camera_rotation_x += dy * 0.5
            
            self.camera_rotation_x = max(-90, min(90, self.camera_rotation_x))
            
            self.last_mouse_pos = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = None
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.camera_distance -= delta * 0.01
        self.camera_distance = max(10, min(50, self.camera_distance))
        self.update()


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fizyka kulek 3D")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        self.gl_widget = GLWidget()
        layout.addWidget(self.gl_widget, stretch=10)  
        
        info_label = QLabel(
            "+/- kulki | ↑/↓ prędkość | SPACJA stop | X/Y/Z oś | L światło | R reset | LPM kamera | Kółko zoom"
        )
        info_label.setStyleSheet("padding: 5px; background-color: #2b2b2b; color: white; font-size: 10pt;")
        layout.addWidget(info_label, stretch=0)  


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
