#shader vertex 
#version 430
layout (location = 0) in vec3 position;

uniform mat4 mv_Matrix;
uniform mat4 proj_Matrix;

out vec4 varyingColor;

void main(void)
{
    gl_Position = proj_Matrix * mv_Matrix * vec4(position,1.0);
    varyingColor = vec4(position,1.0) * 0.5 + vec4(0.5,0.5,0.5,0.5);
}





#shader fragment 
#version 430

out vec4 color;

in vec4 varyingColor;

uniform mat4 mv_Matrix;
uniform mat4 proj_Matrix;


void main(void)
{
    color = varyingColor;
}











