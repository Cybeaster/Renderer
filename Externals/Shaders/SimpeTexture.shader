#shader vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 texPositions;

out vec2 tc;
uniform mat4 mv_matrix;
uniform mat4 proj_matrix;
layout (binding = 0) uniform sampler2D sampler;

void main(void)
{
    gl_Position = proj_matrix * mv_matrix * vec4(position,1.0);
    tc = texPositions;
}


#shader fragment
#version 450

in vec2 tc;
out vec4 color;

uniform mat4 mv_matrix;
uniform mat proj_matrix;
layout (binding = 0) uniform sampler2D sampler;

void main(void)
{
    color texture(sampler,tc);
}