#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

void main()
{
    // Pozycja fragmentu w przestrzeni świata
    vec4 worldPos = model * vec4(position, 1.0);
    FragPos = worldPos.xyz;
    
    // Normalna w przestrzeni świata
    Normal = normalMatrix * normal;
    
    // Współrzędne tekstury
    TexCoord = texCoord;
    
    // Pozycja wierzchołka w przestrzeni clip
    gl_Position = projection * view * worldPos;
}
