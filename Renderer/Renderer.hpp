#pragma once
#include <Test.hpp>
#include "Checks/Assert.hpp"
#include "Math.hpp"
#include "Vector.hpp"
#include "Types.hpp"
#include "UniquePtr.hpp"


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
        static auto getRenderer()
        {
            if (!SingletonRenderer)
               return std::move(SingletonRenderer = TTMakeUnique<Renderer>());
            else
        		return std::move(SingletonRenderer);
        }

        /**
         * @brief Initalizes glfw Opengl context and creates a window.
         *
         * @return GLFWwindow*
         */
        GLFWwindow* GLFWInit();
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
        void GLFWRendererStart(float currentTime);
        void GLFWRendererEnd();
        void CalcDeltaTime(float currentTime);
    	void CleanScene();
        void GLFWCalcPerspective(GLFWwindow *window);

        GLint Height{0};
        GLint Width{0};

        static constexpr uint32 ScreenWidth = 1920;
        static constexpr uint32 ScreenHeight = 1080;

        static float DeltaTime;
        static float LastFrame;
        static float CurrentFrame;
        static TVec3 CameraPos;
        static TMat4 VMat;

        GLFWwindow *Window;
    	TTVector<Test::Test *> Tests;
    	static inline std::unique_ptr<Renderer> SingletonRenderer = nullptr;
    };

}
