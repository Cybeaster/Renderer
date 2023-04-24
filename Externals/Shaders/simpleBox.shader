#shader vertex
#version 450
layout (location=0) in vec3 position;



uniform mat4 proj_matrix;
uniform mat4 mv_matrix;

void main(void)
{
    gl_Position = proj_matrix * mv_matrix * vec4(position,1.f);

}


#shader fragment
#version 450

out vec4 color;

void main(void)
{
    color = vec4(1,1,1,1);
}