#include "SimpleBox.hpp"
#include "Renderer.hpp"

namespace Test
{

    TestSimpleBox::TestSimpleBox(TString shaderPath) : Test(shaderPath)
    {
        AddVertexArray();
        AddBuffer(box, 12);
    }

    void TestSimpleBox::OnUpdate(
        float deltaTime,
        float aspect,
        const TVec3 &cameraPos,
        TMat4 &pMat,
        TMat4 &vMat)
    {
        Test::OnUpdate(deltaTime, aspect, cameraPos, pMat, vMat);

        getShader().SetUnformMat4f("mv_matrix", glm::translate(vMat, TVec3(0.0, 0.0, 0.0)));

        EnableVertexArray(0);

        GLCall(glEnable(GL_DEPTH_TEST));
        GLCall(glFrontFace(GL_CCW));
        GLCall(glDepthFunc(GL_LEQUAL));

        GLCall(glDrawArrays(GL_LINES, 0, 6));
    }
}