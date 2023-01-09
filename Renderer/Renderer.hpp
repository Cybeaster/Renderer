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

        void EnableBuffer(const TBufferAttribVertexHandle &Handle)
        {
            VertexArray.EnableBuffer(Handle);
        }

        void Init();
        void PostInit();

        static inline int ScreenWidth = 1920;
        static inline int ScreenHeight = 1080;

        static inline float Aspect{0};
        static inline float DeltaTime{0};
        static inline float LastFrame{0};
        static inline float CurrentFrame{0};
        static inline float Fovy{1.0472f};
        static inline TVec3 CameraPos{0.f, 1.f, 0.f};

        static inline TMat4 VMat{};
        static inline TMat4 PMat{};

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

        bool PrintFPS = true;

        TVertexArray VertexArray;

        GLFWwindow *Window;
        TTVector<Test::Test *> Tests;
        static inline TTSharedPtr<TRenderer> SingletonRenderer = nullptr;
    };

}
