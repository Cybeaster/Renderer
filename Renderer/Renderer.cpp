
#include "Renderer.hpp"
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
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
namespace RenderAPI
{

    float Renderer::aspect{0};
    glm::mat4 Renderer::pMat{};

    void WindowReshapeCallback(GLFWwindow* window,int newHeight,int newWidth)
    {
        Renderer::aspect = (float)newWidth / (float)newHeight;
        glViewport(0,0,newWidth,newHeight);
        Renderer::pMat = glm::perspective(1.0472f,Renderer::aspect,0.1f,1000.f);
    }

    GLFWwindow* Renderer::Init()
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
        glfwSetWindowSizeCallback(window,WindowReshapeCallback);
        return window;
    }



    Renderer::~Renderer()
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void Renderer::RendererStart(float currentTime)
    {
        CleanScene();
        CalcPerspective(window);
        CalcDeltaTime(currentTime);
    }
    void Renderer::RendererEnd()
    {
        /* Swap front and back buffers */
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    void Renderer::renderTick()
    {
        RendererStart(glfwGetTime());
        
        for(auto* test :tests)
            if(test != nullptr)
                test->OnUpdate(window,deltaTime,aspect,cameraPos,pMat,vMat);
       
        RendererEnd();
    }


    void Renderer::CleanScene()
    {
        GLCall(glClear(GL_COLOR_BUFFER_BIT));
        GLCall(glClear(GL_DEPTH_BUFFER_BIT));
        GLCall(glEnable(GL_CULL_FACE));
    }

    void Renderer::CalcPerspective(GLFWwindow* window)
    {
        glfwGetFramebufferSize(window,&width,&height);
        aspect = float(width) / float(height);
        pMat = glm::perspective(1.0472f,aspect,0.1f,1000.f);
        vMat = glm::translate(glm::mat4(1.0f),cameraPos * -1.f);

    }

    void Renderer::CalcDeltaTime(float currentTime)
    {
        currentFrame = currentTime;
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;        
    }
} // namespace RenderAPI
