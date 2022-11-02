
#include "Renderer.hpp"
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <iostream>

void GLClearError()
{
    while (glGetError() != GL_NO_ERROR)
        ;
}

bool GLLogCall(const char *func, const char *file, int line)
{
    while (GLenum error = glGetError())
    {
        std::cout << "[Opengl Error] (" << std::hex << error << ") :" << func << '\t' << line << '\t' << file << std::endl;
        return false;
    }
    return true;
}
namespace RenderAPI
{
    // All default settings
    TMat4 Renderer::PMat{};
    float Renderer::Aspect{0};
    float Renderer::DeltaTime{0};
    float Renderer::LastFrame{0};
    float Renderer::CurrentFrame{0};
    TVec3 Renderer::CameraPos{0.f, 10.f, 100.f};
    TMat4 Renderer::VMat{};
    Renderer *Renderer::SingletonRenderer = nullptr;

    void WindowReshapeCallback(GLFWwindow *window, int newHeight, int newWidth)
    {
        Renderer::Aspect = (float)newWidth / (float)newHeight;
        glViewport(0, 0, newWidth, newHeight);
        Renderer::PMat = glm::perspective(1.0472f, Renderer::Aspect, 0.1f, 1000.f);
    }

    Renderer::~Renderer()
    {
        glfwDestroyWindow(Window);
        glfwTerminate();
    }

#pragma region GLFW

    GLFWwindow *Renderer::GLFWInit()
    {
        /* Initialize the library */
        assert(glfwInit());

        /* Create a windowed mode window and its OpenGL context */
        Window = glfwCreateWindow(1920, 1080, "Renderer", NULL, NULL);

        if (!Window)
            glfwTerminate();

        /* Make the window's context current */
        glfwMakeContextCurrent(Window);
        glfwSwapInterval(1);

        if (glewInit() != GLEW_OK)
            std::cout << "Error with glewInit()" << std::endl;

        GLCall(glEnable(GL_BLEND));
        GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        glfwSetWindowSizeCallback(Window, WindowReshapeCallback);
        return Window;
    }

    void Renderer::GLFWRendererStart(float currentTime)
    {
        CleanScene();
        GLFWCalcPerspective(Window);
        CalcDeltaTime(currentTime);
    }
    void Renderer::GLFWRendererEnd()
    {
        /* Swap front and back buffers */
        glfwSwapBuffers(Window);
        glfwPollEvents();
    }

    void Renderer::GLFWRenderTickStart()
    {
        while (!glfwWindowShouldClose(Window))
        {
            GLFWRendererStart(glfwGetTime());
            for (auto *test : Tests)
                if (test != nullptr)
                    test->OnUpdate(DeltaTime, Aspect, CameraPos, PMat, VMat);
            GLFWRendererEnd();
        }
    }

    void Renderer::GLFWCalcPerspective(GLFWwindow *window)
    {
        glfwGetFramebufferSize(window, &Width, &Height);
        Aspect = float(Width) / float(Height);
        PMat = glm::perspective(1.0472f, Aspect, 0.1f, 1000.f);
        VMat = glm::translate(TMat4(1.0f), CameraPos * -1.f);
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
        CurrentFrame = currentTime;
        DeltaTime = CurrentFrame - LastFrame;
        LastFrame = CurrentFrame;
    }

    void Renderer::AddTest(Test::Test *testPtr)
    {
        if (testPtr != nullptr)
        {
            testPtr->Init(PMat);
            Tests.push_back(testPtr);
        }
    }
} // namespace RenderAPI
