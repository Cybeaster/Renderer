#include "SimpleBox.hpp"
#include "Renderer.hpp"
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

namespace test
{

    TestSimpleBox::TestSimpleBox(std::string shaderPath) : Test(shaderPath)
    {
        AddVertexArray();
        AddBuffer(box,12);
    }


    void TestSimpleBox::OnUpdate(
        float deltaTime,
        float aspect,
        const glm::vec3& cameraPos,
        glm::mat4& pMat,
        glm::mat4& vMat)
        {
            Test::OnUpdate(deltaTime,aspect,cameraPos,pMat,vMat);
            
    
            getShader().SetUnformMat4f("mv_matrix",glm::translate(vMat,glm::vec3(0.0,0.0,0.0)));

            
            EnableVertexArray(0);

            GLCall(glEnable(GL_DEPTH_TEST));
            GLCall(glFrontFace(GL_CCW));
            GLCall(glDepthFunc(GL_LEQUAL));

            GLCall(glDrawArrays(GL_LINES,0,6));
        }
}