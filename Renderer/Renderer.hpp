#pragma once
#include <Test.hpp>
#include "Checks/Assert.hpp"
#include "Math.hpp"
#include "Vector.hpp"
#include "Types.hpp"
#include "SmartPtr.hpp"
#include "Vertex/VertexArray.hpp"
#include "ThreadPool.hpp"

#define GLCall(x)   \
    GLClearError(); \
    x;              \
    ASSERT(GLLogCall(#x, __FILE__, __LINE__))

void GLClearError();
bool GLLogCall(const char *func, const char *file, int line);

struct GLFWwindow;
class Application;
namespace RenderAPI
{
    /**
     * @brief Singleton class that creates the context, calculates perspective, frames etc.
     *
     */
    class TRenderer
    {
    public:
        static auto GetRenderer()
        {
            if (!SingletonRenderer)
            {
                SingletonRenderer = TTSharedPtr<TRenderer>(new TRenderer());
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

        ~TRenderer();

        TDrawVertexHandle CreateVertexElement(const TVertexContext &VContext, const TDrawContext &RContext)
        {
            return VertexArray.CreateVertexElement(VContext, RContext);
        }

        void DrawArrays(const TDrawVertexHandle &Handle)
        {
            VertexArray.DrawArrays(Handle);
        }

        void EnableBuffer(const TDrawVertexHandle &Handle)
        {
            VertexArray.EnableBuffer(Handle);
        }

        void EnableBuffer(const TBufferAttribVertexHandle &Handle)
        {
            VertexArray.EnableBuffer(Handle);
        }

        void Init();
        void PostInit();

        static int ScreenWidth;
        static int ScreenHeight;

        static float Aspect;
        static float DeltaTime;
        static float LastFrame;
        static float CurrentFrame;
        static float Fovy;
        static TVec3 CameraPos;

        static TMat4 VMat;
        static TMat4 PMat;

        void TranslateCameraLocation(const glm::mat4 &Transform);

        void LookAtCamera(const TVec3 &Position);

        TBufferAttribVertexHandle AddAttribBuffer(const TVertexAttribBuffer &Buffer)
        {
            return VertexArray.AddAttribBuffer(Buffer);
        }

        TBufferAttribVertexHandle AddAttributeBuffer(const TVertexContext &Context)
        {
            return VertexArray.AddAttribBuffer(Context);
        }

    private:
        TRenderer() = default;
        Thread::TThreadPool RendererThreadPool;

        void GLFWRendererStart(float currentTime);
        void GLFWRendererEnd();
        void CalcDeltaTime(float currentTime);
        void CleanScene();
        void GLFWCalcPerspective(GLFWwindow *window);
        void PrintDebugInfo();

        bool PrintFPS = false;

        TVertexArray VertexArray;

        GLFWwindow *Window;
        TTVector<Test::Test *> Tests;
        static inline TTSharedPtr<TRenderer> SingletonRenderer = nullptr;
    };

}
