#include "Renderer.hpp"
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <iostream>

void GLClearError()
{
    while (glGetError() != GL_NO_ERROR)
        ;
}

bool GLLogCall(const char *func, const char *file, const int line)
{
    while (const GLenum error = glGetError())
    {
        std::cout << "[Opengl Error] (" << std::hex << error << ") :" << func << '\t' << line << '\t' << file << std::endl;
        return false;
    }
    return true;
}
namespace RenderAPI
{

    // std::unique_ptr<Renderer> Renderer::SingletonRenderer = nullptr;

    void WindowReshapeCallback(GLFWwindow *window, const int newHeight, const int newWidth)
    {
        if (!window)
            return;
        TRenderer::Aspect = static_cast<float>(newWidth / newHeight);
        glViewport(0, 0, newWidth, newHeight);
        TRenderer::PMat = glm::perspective(TRenderer::Fovy, TRenderer::Aspect, 0.1f, 1000.f);
    }

    TRenderer::~TRenderer()
    {
        glfwDestroyWindow(Window);
        glfwTerminate();
    }

    void TRenderer::Init()
    {

        // Post Init has to be called after everything
        PostInit();
    }

    void TRenderer::PostInit()
    {
        VertexArray.AddVertexArray();
    }
#pragma region GLFW

    GLFWwindow *
    TRenderer::GLFWInit()
    {
        /* Initialize the library */
        assert(glfwInit());

        /* Create a windowed mode window and its OpenGL context */
        Window = glfwCreateWindow(ScreenWidth, ScreenHeight, "Renderer", NULL, NULL);

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

    void TRenderer::GLFWRendererStart(const float currentTime)
    {
        CleanScene();
        GLFWCalcPerspective(Window);
        CalcDeltaTime(currentTime);
        PrintDebugInfo();
    }
    void TRenderer::GLFWRendererEnd()
    {
        /* Swap front and back buffers */
        glfwSwapBuffers(Window);
        glfwPollEvents();
    }

    void TRenderer::GLFWRenderTickStart()
    {
        while (!glfwWindowShouldClose(Window))
        {
            GLFWRendererStart(glfwGetTime());
            for (auto *const test : Tests)
                if (test)
                    test->OnUpdate(DeltaTime, Aspect, CameraPos, PMat, VMat);
            GLFWRendererEnd();
        }
    }
    void TRenderer::PrintDebugInfo()
    {
        if (PrintFPS)
        {
            std::cout << "Current FPS is " << 1 / DeltaTime << '\n';
        }
    }

    void TRenderer::GLFWCalcPerspective(GLFWwindow *window)
    {
        glfwGetFramebufferSize(window, &ScreenWidth, &ScreenHeight);
        Aspect = static_cast<float>(ScreenWidth / ScreenHeight);
        PMat = glm::perspective(1.0472f, Aspect, 0.1f, 1000.f);
        VMat = glm::translate(TMat4(1.0f), CameraPos * -1.f);
    }

    void TRenderer::TranslateCameraLocation(const glm::mat4 &Transform)
    {
        //CameraPos *= Transform;
    }

    void TRenderer::LookAtCamera(const TVec3 &Position)
    {
        //CameraPos *= glm::lookAt(CameraPos,Position,TVec3(0,0,1));
    }

#pragma endregion GLFW

    void TRenderer::CleanScene()
    {
        GLCall(glClear(GL_COLOR_BUFFER_BIT));
        GLCall(glClear(GL_DEPTH_BUFFER_BIT));
        GLCall(glEnable(GL_CULL_FACE));
    }

    void TRenderer::CalcDeltaTime(const float currentTime)
    {
        CurrentFrame = currentTime;
        DeltaTime = CurrentFrame - LastFrame;
        LastFrame = CurrentFrame;
    }

    void TRenderer::AddTest(Test::Test *testPtr)
    {
        if (testPtr)
        {
            testPtr->Init(PMat);
            Tests.push_back(testPtr);
        }
    }
} // namespace RenderAPI
