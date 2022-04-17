#pragma once
#include <glm.hpp>
#include "Shader.hpp"
#include "glfw3.h"

#define ASSERT(x) if ((!x)) __debugbreak();

#define GLCall(x) GLClearError();\
    x;\
    ASSERT(GLLogCall(#x,__FILE__,__LINE__))


void GLClearError();
bool GLLogCall(const char* func, const char* file, int line);

namespace RenderAPI
{
    class Renderer 
    {

    public:

        void RenderTick());
        GLFWwindow* Init(const std::string shaderSource);


        Renderer() = default;
        ~Renderer();

    protected:


    private:

        void RendererStart(float currentTime);
        void RendererEnd();
        void CalcDeltaTime(float currentTime);
        void CleanScene();
        void CalcPerspective(GLFWwindow* window);

        glm::mat4 
        pMat, // perspective matrix
        vMat, // view matrix
        mMat, // model matrix
        mvMat, //model-view matrix
        tMat, //
        rMat;

        float aspect{0.f};

        GLint height{0};
        GLint width{0};

        Shader shader;

        float deltaTime = 0.f;

        float lastFrame = 0.f;
        float currentFrame = 0.f;

        GLFWwindow* window;
    };

} 

