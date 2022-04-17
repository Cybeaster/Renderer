#pragma once
#define GLEW_STATIC

#include "GL/glew.h"
#include <VertexArray.hpp>
#include <VertexBuffer.hpp>
#include <Shader.hpp>
#include <IndexBuffer.hpp>

#define ASSERT(x) if ((!x)) __debugbreak();

#define GLCall(x) GLClearError();\
    x;\
    ASSERT(GLLogCall(#x,__FILE__,__LINE__))


void GLClearError();
bool GLLogCall(const char* func, const char* file, int line);

class Renderer
{
    
public:
    Renderer();
    ~Renderer();


    void Draw();
    void Init();     
    void Clear() const;

private:


};




