#pragma once
#include <Test.hpp>
#include "Checks/Assert.hpp"
#include "Math.hpp"
#include "Vector.hpp"
#include "Types.hpp"
#include "SmartPtr.hpp"
#include "ThreadPool.hpp"

#define GLCall(x)   \
    GLClearError(); \
    x;              \
    ASSERT(GLLogCall(#x, __FILE__, __LINE__))

void GLClearError();
bool GLLogCall(const char *func, const char *file, int line);

class GLFWwindow;
class Application;
namespace RenderAPI
{
    /**
     * @brief Singleton class that creates the context, calculates perspective, frames etc.
     *
     */
    class Renderer
    {
    public:
        static auto GetRenderer()
        {
            if (!SingletonRenderer)
            {
                SingletonRenderer = TTSharedPtr<Renderer>(new Renderer());
                return SingletonRenderer;
            }
            else
            {
                return SingletonRenderer;
            }
        }

        /**
         * @brief Initalizes glfw Opengl context and creates a window.
         *
         * @return GLFWwindow*
         */
        GLFWwindow *GLFWInit();
        void GLFWRenderTickStart();

        void AddTest(Test::Test *testPtr);

        inline TTVector<Test::Test *> &getTests()
        {
            return Tests;
        }

        static float Aspect;
        static TMat4 PMat;

        ~Renderer();

    private:

        Renderer() = default;
        Thread::TThreadPool RendererThreadPool;

        void GLFWRendererStart(float currentTime);
        void GLFWRendererEnd();
        void CalcDeltaTime(float currentTime);
        void CleanScene();
        void GLFWCalcPerspective(GLFWwindow *window);
        void PrintDebugInfo();
        
        GLint Height{0};
        GLint Width{0};

        bool PrintFPS = true;

        static constexpr uint32 ScreenWidth = 1920;
        static constexpr uint32 ScreenHeight = 1080;

        static float DeltaTime;
        static float LastFrame;
        static float CurrentFrame;
        static TVec3 CameraPos;
        static TMat4 VMat;

        GLFWwindow *Window;
        TTVector<Test::Test *> Tests;
        static inline TTSharedPtr<Renderer> SingletonRenderer = nullptr;
    };

}
