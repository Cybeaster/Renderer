#shader vertex
#version 330 core 
layout(location = 0) in vec4 position;

uniform mat4 mv_Matrix;
uniform vec4 proj_Matrix;


void main()
{
    gl_Position = mv_Matrix * position * proj_Matrix; 
};




#shader fragment
#version 330 core  

layout(location = 0) out vec4 color;

uniform vec4 u_Color;

void main()
{
   color = u_Color; 
};