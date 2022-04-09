#pragma once
#include <Test.hpp>
#include <Renderer.hpp>
#include <cstdint>
#include <glm/glm.hpp>
#include "GLFW/glfw3.h"
namespace test
{
    class TestSimpleCube
    {
    public:

        TestSimpleCube(/* args */);
        ~TestSimpleCube();

        void OnRender(GLFWwindow* window,float currentTime);
        
        GLuint renderingProg = 0;
    private:
        
        float vertexPositions[108] = {
        -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
        1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
        1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
        1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f
        };

        Shader shader{"../../Externals/Shaders/SimpleCube.shader"};
        GLuint mvLoc;
        GLuint projLocation;
        GLuint vao[1];
        GLuint vbo[2];
        int32_t width, height;
        glm::mat4 pMat,vMat,mMat,mvMat,tMat,rMat;
        glm::vec3 cameraPos{0.0f,0.0f,8.0f};
        glm::vec3 cubePos{0.0f,-2.0f,0.0f};
        float aspect;
    };
    
    
} // namespace test
