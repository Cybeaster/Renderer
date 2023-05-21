#shader vertex
#version 430

layout(location = 0) in vec3 vertexPosition;
uniform mat4 shadowMVP;

void main() {
    gl_Position = shadowMVP * vec4(vertexPosition, 1.0);
}



#shader fragment
#version 430
void main()
{
}

