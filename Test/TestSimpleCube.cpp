#include <TestSimpleCube.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include "glfw3.h"

namespace Test
{

    void TestSimpleCube::OnUpdate(
        float deltaTime,
        float aspect,
        const TVec3 &cameraPos,
        TMat4 &pMat,
        TMat4 &vMat)
    {
        Test::OnUpdate(deltaTime, aspect, cameraPos, pMat, vMat);

        GetMVStack().push(vMat);
        GetMVStack().push(GetMVStack().top());

        GetMVStack().top() *= glm::translate(TMat4(1.0f), TVec3(0.0, 0.0, 0.0));
        GetMVStack().push(GetMVStack().top());
        GetMVStack().top() *= glm::rotate(TMat4(1.0f), float(deltaTime), TVec3(1.0, 0.0, 0.0));
        getShader().SetUnformMat4f("mv_matrix", GetMVStack().top());

        EnableVertexArray(1);
        GLCall(glEnable(GL_DEPTH_TEST));
        GLCall(glFrontFace(GL_CCW));
        GLCall(glDepthFunc(GL_LEQUAL));
        GLCall(glDrawArrays(GL_TRIANGLES, 0, 18));
        GetMVStack().pop();
        // pyr

        // cube

        GetMVStack().push(GetMVStack().top());
        GetMVStack().top() *= glm::translate(TMat4(1.0f), TVec3(sin(float(deltaTime)) * 4.0f, 0.0f, cos(float(deltaTime) * 4.0)));
        GetMVStack().push(GetMVStack().top());
        GetMVStack().top() *= glm::rotate(TMat4(1.0f), float(deltaTime), TVec3(0.0, 1.0, 0.0));
        getShader().SetUnformMat4f("mv_matrix", GetMVStack().top());

        EnableVertexArray(1);
        GLCall(glEnable(GL_DEPTH_TEST));
        GLCall(glFrontFace(GL_CCW));
        GLCall(glDepthFunc(GL_LEQUAL));
        GLCall(glDrawArrays(GL_TRIANGLES, 0, 36));
        GetMVStack().pop();

        // smaller cube

        GetMVStack().push(GetMVStack().top());
        GetMVStack().top() *= glm::translate(TMat4(1.0f), TVec3(0.f, sin(deltaTime) * 2.0, cos(float(deltaTime) * 2.0)));
        GetMVStack().top() *= glm::rotate(TMat4(1.0f), float(deltaTime), TVec3(0.0, 0.0, 1.0));
        GetMVStack().top() *= glm::scale(TMat4(1.0), TVec3(0.25f, 0.25f, 0.25f));
        getShader().SetUnformMat4f("mv_matrix", GetMVStack().top());

        EnableVertexArray(1);
        GLCall(glEnable(GL_DEPTH_TEST));
        GLCall(glFrontFace(GL_CW));
        GLCall(glDepthFunc(GL_LEQUAL));
        GLCall(glDrawArrays(GL_TRIANGLES, 0, 36));
        GetMVStack().pop();
        GetMVStack().pop();
        GetMVStack().pop();
        GetMVStack().pop();
    }

    TestSimpleCube::TestSimpleCube(TString shaderPath) : Test(shaderPath)
    {
        AddVertexArray();
        AddBuffer(vertexPositions, sizeof(vertexPositions));
        AddBuffer(pyramidPositions, sizeof(pyramidPositions));
    }

    TestSimpleCube::~TestSimpleCube()
    {
    }

}
