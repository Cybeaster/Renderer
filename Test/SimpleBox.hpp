#pragma once

#include "Test.hpp"

namespace test
{
    class TestSimpleBox : public Test
    {
        
    public:
        TestSimpleBox() = default;
        TestSimpleBox(std::string shaderPath);


        void OnUpdate(GLFWwindow* window,
        float deltaTime,
        float aspect,
        const glm::vec3& cameraPos,
        glm::mat4& pMat,
        glm::mat4& vMat) override;
        
        float box[18]
        {
            0.f,1.f,1.f,
            1.f,1.f,1.f,
        };
    };
    
 
}