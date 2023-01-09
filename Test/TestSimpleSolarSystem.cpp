
#include <TestSimpleSolarSystem.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include "glfw3.h"

namespace Test
{

    void TestSimpleSolarSystem::OnUpdate(
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
        GetShader().SetUnformMat4f("mv_matrix", GetMVStack().top());

        DrawArrays(pyramidHandle);

        GetMVStack().pop();
        // pyr

        // cube

        GetMVStack().push(GetMVStack().top());
        GetMVStack().top() *= glm::translate(TMat4(1.0f), TVec3(sin(float(deltaTime)) * 4.0f, 0.0f, cos(float(deltaTime) * 4.0)));
        GetMVStack().push(GetMVStack().top());
        GetMVStack().top() *= glm::rotate(TMat4(1.0f), float(deltaTime), TVec3(0.0, 1.0, 0.0));
        GetShader().SetUnformMat4f("mv_matrix", GetMVStack().top());

        DrawArrays(cubeHandle);

        GetMVStack().pop();

        // smaller cube

        GetMVStack().push(GetMVStack().top());
        GetMVStack().top() *= glm::translate(TMat4(1.0f), TVec3(0.f, sin(deltaTime) * 2.0, cos(float(deltaTime) * 2.0)));
        GetMVStack().top() *= glm::rotate(TMat4(1.0f), float(deltaTime), TVec3(0.0, 0.0, 1.0));
        GetMVStack().top() *= glm::scale(TMat4(1.0), TVec3(0.25f, 0.25f, 0.25f));
        GetShader().SetUnformMat4f("mv_matrix", GetMVStack().top());

        DrawArrays(cubeHandle);

        GetMVStack().pop();
        GetMVStack().pop();
        GetMVStack().pop();
        GetMVStack().pop();
    }

    TestSimpleSolarSystem::TestSimpleSolarSystem(TPath shaderPath, TRenderer *Renderer) : Test(shaderPath, Renderer)
    {
        auto size = sizeof(cubePositions);
        auto data = cubePositions;
        TVertexContext contextVertex(new TBuffer{data, size}, 0, 3, GL_FLOAT, false, 0);

        TDrawContext drawContext(GL_TRIANGLES,
                                 0,
                                 size / 3,
                                 GL_LEQUAL,
                                 GL_CCW,
                                 GL_DEPTH_TEST,
                                 0);
        cubeHandle = CreateVertexElement(contextVertex, drawContext);

        size = sizeof(pyramidPositions);
        data = pyramidPositions;

        TVertexContext pyramidVertex(new TBuffer{pyramidPositions, sizeof(pyramidPositions)}, 0, 3, GL_FLOAT, false, 0);
        
        TDrawContext pyramidDrawContext(GL_TRIANGLES,
                                        0,
                                        size / 3,
                                        GL_LEQUAL,
                                        GL_CCW,
                                        GL_DEPTH_TEST,
                                        0);

        pyramidHandle = CreateVertexElement(pyramidVertex, pyramidDrawContext);
    }

}
