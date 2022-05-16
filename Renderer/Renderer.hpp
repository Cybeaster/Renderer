#pragma once
#include <glm.hpp>
#include <Test.hpp>

#define ASSERT(x) if ((!x)) __debugbreak();

#define GLCall(x) GLClearError();\
    x;\
    ASSERT(GLLogCall(#x,__FILE__,__LINE__))


void GLClearError();
bool GLLogCall(const char* func, const char* file, int line);


class GLFWwindow;
namespace RenderAPI
{
    class Renderer 
    {

    public:

        void renderTick();

        /**
         * @brief Инициализирует контекст Opengl и создает окно.
         * 
         * @return GLFWwindow* 
         */
        GLFWwindow* Init();

        
        void addTest(test::Test* testPtr)
        {
            if(testPtr != nullptr)
            {
                testPtr->Init(pMat);
                tests.push_back(testPtr);
            }
        
        }

        Renderer() = default;
        ~Renderer();

        static float aspect;
        static glm::mat4 pMat;
    private:

        void RendererStart(float currentTime);
        void RendererEnd();
        void CalcDeltaTime(float currentTime);
        void CleanScene();
        void CalcPerspective(GLFWwindow* window);

        glm::mat4 vMat; // view matrix
 
        GLint height{0};
        GLint width{0};

        float deltaTime = 0.f;
        float lastFrame = 0.f;
        float currentFrame = 0.f;

        GLFWwindow* window;

        glm::vec3 cameraPos{0.f,2.f,100.f};

        std::vector<test::Test*> tests;
    };

} 

