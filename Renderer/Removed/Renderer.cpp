#include "Renderer.hpp"
#include <string>
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


void Renderer::Draw()
{
    

}

void Renderer::Clear() const 
{
    GLCall(glClear(GL_COLOR_BUFFER_BIT));
}

void Renderer:: Init()
{
    
    
}

Renderer::Renderer()
{

}

Renderer::~Renderer()
{

}