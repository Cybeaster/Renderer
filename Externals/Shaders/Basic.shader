#shader vertex

#version 430
layout(location=0) in vec3 position;
layout(location=1) in vec2 texels;
layout(location=2) in vec3 normals;

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



out vec2 tc;

uniform mat4 proj_matrix;
uniform mat4 mv_matrix;
uniform int use_texture = 1;


uniform PositionalLight light;
uniform Material material;
uniform mat4 norm_matrix;

uniform vec4 globalAmbient;

layout (binding = 0) uniform sampler2D sampler;

void main(void)
{

    vec4 color;

    gl_Position = proj_matrix * mv_matrix * vec4(position,1.0);
    tc = texels;
}




#shader fragment
#version 430

in vec2 tc;
out vec4 color;

layout (binding=0) uniform sampler2D sampler;
uniform mat4 proj_matrix;
uniform mat4 mv_matrix;
uniform int use_texture = 1;

void main(void)
{
    color = vec4(1,1,1,1);
   if(use_texture == 1)
   {
        color = texture(sampler,tc);
    }
};