#include "Renderer.hpp"
#include <iostream>

void GLClearError()
{
    while(glGetError() != GL_NO_ERROR);
}

bool GLLogCall(const char* func, const char* file, int line)
{
    while(GLenum error = glGetError())
    {
        std::cout<< "[Opengl Error] (" << std::hex<< error << ") :" << func <<'\t'<< line <<'\t'<< file<< std::endl;
        return false;
    }
    return true;
}


void Renderer::Draw(const VertexArray& va, const IndexBuffer& ib, const Shader& shader)
{
    shader.Bind();
    va.Bind();
    ib.Bind(); 
    GLCall(glDrawElements(GL_TRIANGLES,ib.GetCount(),GL_UNSIGNED_INT,nullptr));

}
void Renderer::Clear() const 
{
    GLCall(glClear(GL_COLOR_BUFFER_BIT));
}

Renderer::Renderer()
{
}

Renderer::~Renderer()
{

}