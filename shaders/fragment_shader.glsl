#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

// Światła (maksymalnie 8, jak w OpenGL fixed-function pipeline)
#define MAX_LIGHTS 8

uniform int numLights;
uniform vec3 lightPositions[MAX_LIGHTS];
uniform vec3 lightColors[MAX_LIGHTS];

uniform vec3 viewPos;
uniform vec3 materialColor;
uniform sampler2D texture1;
uniform bool useTexture;
uniform float shininess;

void main()
{
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    // Wynik oświetlenia
    vec3 result = vec3(0.0);
    
    // Dla każdego światła
    for(int i = 0; i < numLights && i < MAX_LIGHTS; i++) {
        vec3 lightColor = lightColors[i];
        
        // Ambient
        float ambientStrength = 0.3;
        vec3 ambient = ambientStrength * lightColor;
        
        // Diffuse
        vec3 lightDir = normalize(lightPositions[i] - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        // Specular (Phong)
        float specularStrength = 1.0;
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        vec3 specular = specularStrength * spec * lightColor;
        
        // Attenuation (osłabienie z odległością)
        float distance = length(lightPositions[i] - FragPos);
        float attenuation = 1.0 / (1.0 + 0.05 * distance + 0.01 * distance * distance);
        
        // Suma dla tego światła
        result += (ambient + diffuse + specular) * attenuation;
    }
    
    // Zastosuj kolor materiału lub teksturę
    if (useTexture) {
        vec3 texColor = texture(texture1, TexCoord).rgb;
        result *= texColor;
    } else {
        result *= materialColor;
    }
    
    FragColor = vec4(result, 1.0);
}

