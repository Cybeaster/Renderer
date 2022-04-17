#include "Renderer.hpp"
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include "GL/glew.h"
#include <iostream>

namespace RenderAPI
{

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

    GLFWwindow* Renderer::Init(const std::string shaderSource)
    {
        
        /* Initialize the library */
        assert(glfwInit());
        
        /* Create a windowed mode window and its OpenGL context */
         window = glfwCreateWindow(1920, 1080, "Renderer", NULL, NULL);

        if (!window)
            glfwTerminate();
    
        /* Make the window's context current */
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        if(glewInit() != GLEW_OK)
            std::cout<<"Error with glewInit()"<<std::endl;
    
        GLCall(glEnable(GL_BLEND));
        GLCall(glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA));

        shader.Init(shaderSource);
    }



    Renderer::~Renderer()
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void Renderer::RendererStart(float currentTime)
    {
        CleanScene();
        CalcDeltaTime(currentTime);

        GLCall(glEnable(GL_CULL_FACE));
        CalcPerspective(window);
        shader.Bind();
    }
    void Renderer::RendererEnd()
    {
        /* Swap front and back buffers */
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    void Renderer::RenderTick()
    {
        RendererStart(glfwGetTime());
        

        RendererEnd();
    }


    void Renderer::CleanScene()
    {
        GLCall(glClear(GL_COLOR_BUFFER_BIT));
        GLCall(glClear(GL_DEPTH_BUFFER_BIT));
    }

    void Renderer::CalcPerspective(GLFWwindow* window)
    {
        glfwGetFramebufferSize(window,&width,&height);
        aspect = float(width) / float(height);
        pMat = glm::perspective(1.0472f,aspect,0.1f,1000.f);
    }

    void Renderer::CalcDeltaTime(float currentTime)
    {
        currentFrame = currentTime;
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;        
    }
} // namespace RenderAPI
