#shader vertex
#version 330 core 

layout(location = 0) in vec4 position;

mat4 buildRotate(float rad)
{
    mat4 xrot = mat4(1.0,0.0,0.0,0.0,
                    0.0,cos(rad),-sin(rad),0.0,
                    0.0,sin(rad),cos(rad),0.0,
                    0.0,0.0,0.0,1.0);
    return xrot;
}

void main()
{
    gl_Position = buildRotate(50) * position; 

};


#shader fragment
#version 330 core  

layout(location = 0) out vec4 color;

in vec2 v_TexCoord;

uniform vec4 u_Color;

void main()
{
   color = u_Color; 
};