#shader vertex
#version 430
layout(location=0) in vec3 vertPosition;
layout(location=1) in vec2 texels;
layout(location=2) in vec3 normal;

out vec3 varyingNormal;
out vec3 varyingLightDir;
out vec3 varyingHalfVector;



out vec3 varyingVertPos;
out vec2 tc;

struct PositionalLight
{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    vec3 position;
};

struct Material
{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
};


uniform mat4 proj_matrix;
uniform mat4 mv_matrix;

uniform PositionalLight light;

uniform Material material;

uniform mat4 norm_matrix;
uniform vec4 globalAmbient;



uniform int use_texture = 1;

layout (binding = 0) uniform sampler2D sampler;



void main(void)
{
    varyingVertPos = (mv_matrix * vec4(vertPosition, 1.0)).xyz;

    varyingLightDir = light.position - varyingVertPos;
    varyingHalfVector = (varyingLightDir + (-varyingVertPos)).xyz;


    varyingNormal = (norm_matrix * vec4(normal, 1.0)).xyz;
    gl_Position = proj_matrix * mv_matrix * vec4(vertPosition, 1.0);
    tc = texels;
}




#shader fragment
#version 430

in vec2 tc;
in vec3 varyingNormal;
in vec3 varyingLightDir;
in vec3 varyingHalfVector;
in vec3 varyingVertPos;

out vec4 fragColor;

layout (binding=0) uniform sampler2D sampler;
uniform mat4 proj_matrix;
uniform mat4 mv_matrix;
uniform mat4 norm_matrix;

uniform int use_texture = 1;


struct PositionalLight
{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    vec3 position;
};

struct Material
{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
};

uniform vec4 globalAmbient;

uniform PositionalLight light;

uniform Material material;


void main(void)
{
    vec3 L = normalize(varyingLightDir);
    vec3 V = normalize(-varyingVertPos);
    vec3 N = normalize(varyingNormal);
    vec3 H = normalize(varyingHalfVector);

    float cosTheta = dot(L, N);
    float cosPhi = dot(H, N);


    if (use_texture == 1)
    {
        vec4 texColor = texture(sampler, tc);

        fragColor = texColor * (globalAmbient + light.ambient * material.ambient  + light.diffuse * material.diffuse * max(cosTheta, 0.0)) + light.specular * material.specular * pow(max(cosPhi, 0.0), material.shininess * 3.0);
    }
    else
    {

        vec4 ambient = ((globalAmbient * material.ambient) + (light.ambient * material.ambient));
        vec4 diffuse = light.diffuse * material.diffuse * max(cosTheta, 0.0);
        vec4 specular = light.specular * material.specular * pow(max(cosPhi, 0.0), material.shininess * 3.0);

        fragColor = (ambient + diffuse + specular);
    }
};