#shader vertex
#version 430
layout (location=0) in vec3 pos;
layout (location=1) in vec2 texCoord;
out vec2 tc; // texture coordinate output to rasterizer for interpolation
uniform mat4 mv_matrix;
uniform mat4 proj_matrix;
layout (binding=0) uniform sampler2D samp; // not used in vertex shader
void main(void)
{ gl_Position = proj_matrix * mv_matrix * vec4(pos,1.0);
tc = texCoord;
}


#shader fragment
#version 430
in vec2 tc; // interpolated incoming texture coordinate
out vec4 color;
uniform mat4 mv_matrix;
uniform mat4 proj_matrix;
layout (binding=0) uniform sampler2D samp;
void main(void)
{ color = texture(samp, tc);
}