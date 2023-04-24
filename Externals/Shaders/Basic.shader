#shader vertex

#version 430
layout(location=0) in vec3 position;
layout(location=1) in vec2 texels;
layout(location = 2) in vec3 normals;

out vec2 tc;

uniform mat4 proj_matrix;
uniform mat4 mv_matrix;
uniform int use_texture = 1;

layout (binding = 0) uniform sampler2D sampler;

void main(void)
{
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