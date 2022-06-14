#pragma once
#include <glm.hpp>
#include <Test.hpp>
#include <memory.h>

#define ASSERT(x) if ((!x)) __debugbreak();

#define GLCall(x) GLClearError();\
    x;\
    ASSERT(GLLogCall(#x,__FILE__,__LINE__))


void GLClearError();
bool GLLogCall(const char* func, const char* file, int line);


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

        static Renderer* getRenderer()
        {
            if(renderer == nullptr)
            {
                renderer = new Renderer();
                return renderer;
            }
            else
                return renderer;
        }

        /**
         * @brief Initalizes glfw Opengl context and creates a window.
         * 
         * @return GLFWwindow* 
         */
        GLFWwindow* GLFWInit();
        void GLFWRenderTickStart();
        

        void addTest(test::Test* testPtr);

        inline std::vector<test::Test*>& getTests()
        {return tests;}
        

        static float aspect;
        static glm::mat4 pMat;

         ~Renderer();
    private:
    
        void GLFWRendererStart(float currentTime);
        void GLFWRendererEnd();
        void CalcDeltaTime(float currentTime);
        void CleanScene();
        void GLFWCalcPerspective(GLFWwindow* window);
 
        GLint height{0};
        GLint width{0};

        uint32_t screenWidth = 1920;
        uint32_t screenHeight = 1080;

        static float deltaTime;
        static float lastFrame;
        static float currentFrame;
        static glm::vec3 cameraPos;
        static glm::mat4 vMat;

        GLFWwindow* window;
        std::vector<test::Test*> tests;
        static Renderer* renderer;
    };

} 

