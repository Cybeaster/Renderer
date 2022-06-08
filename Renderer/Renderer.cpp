
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
    float Renderer::deltaTime{0};
    float Renderer::lastFrame{0};
    float Renderer::currentFrame{0};
    glm::vec3 Renderer::cameraPos{0.f,2.f,100.f};
    glm::mat4 Renderer::vMat{};
    Renderer* Renderer::renderer = nullptr;





    void WindowReshapeCallback(GLFWwindow* window,int newHeight,int newWidth)
    {
        Renderer::aspect = (float)newWidth / (float)newHeight;
        glViewport(0,0,newWidth,newHeight);
        Renderer::pMat = glm::perspective(1.0472f,Renderer::aspect,0.1f,1000.f);
    }

    Renderer::~Renderer()
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }


#pragma region GLFW

    GLFWwindow* Renderer::GLFWInit()
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

    void Renderer::GLFWRendererStart(float currentTime)
    {
        CleanScene();
        GLFWCalcPerspective(window);
        CalcDeltaTime(currentTime);
    }
    void Renderer::GLFWRendererEnd()
    {
        /* Swap front and back buffers */
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    void Renderer::GLFWRenderTickStart()
    {
        while (!glfwWindowShouldClose(window))
        {
            GLFWRendererStart(glfwGetTime());
            for(auto* test :tests)
                if(test != nullptr)
                    test->OnUpdate(deltaTime,aspect,cameraPos,pMat,vMat);
            if(glfwGetTime() > 30.f)
            {
                for(auto* test : tests)
                    test->OnTestEnd();
                exit(0);
            }
            GLFWRendererEnd();
        }
        
    }

    void Renderer::GLFWCalcPerspective(GLFWwindow* window)
    {
        glfwGetFramebufferSize(window,&width,&height);
        aspect = float(width) / float(height);
        pMat = glm::perspective(1.0472f,aspect,0.1f,1000.f);
        vMat = glm::translate(glm::mat4(1.0f),cameraPos * -1.f);
    }
#pragma endregion GLFW

    

    void Renderer::CleanScene()
    {
        GLCall(glClear(GL_COLOR_BUFFER_BIT));
        GLCall(glClear(GL_DEPTH_BUFFER_BIT));
        GLCall(glEnable(GL_CULL_FACE));
    }

    

    void Renderer::CalcDeltaTime(float currentTime)
    {
        currentFrame = currentTime;
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;        
    }

    void Renderer::addTest(test::Test* testPtr)
        {
            if(testPtr != nullptr)
            {
                testPtr->Init(pMat);
                tests.push_back(testPtr);
            }
        }
} // namespace RenderAPI
