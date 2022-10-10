#include "Test.hpp"
#include <Renderer.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
namespace Test
{

    Test::Test(String shaderPath) : shader(shaderPath)
    {
    }

    Test::~Test()
    {
    }

    void Test::AddVertexArray()
    {
        vertexArray.push_back({});
        GLuint *vaID = &vertexArray[vertexArray.size() - 1];

        GLCall(glGenVertexArrays(1, vaID));
        GLCall(glBindVertexArray(*vaID));
    }

    void Test::AddBuffer(void *buffer, int32_t size)
    {
        buffers.push_back(std::make_shared<VertexBuffer>(buffer, size));
    }

    void Test::AddBuffers(Vector<Vector<float>> &vertecis, size_t numOfBuffers)
    {
        for (size_t i = 0; i < numOfBuffers; i++)
        {
        }
    }

    void Test::InitShader(String shaderPath)
    {
        shader.Init(shaderPath);
    }

    void Test::EnableVertexArray(GLuint bufferID)
    {
        buffers[bufferID]->Bind();
        GLCall(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0));
        GLCall(glEnableVertexAttribArray(0));
    }

    void Test::OnUpdate(
        const float deltaTime,
        const float aspect,
        const Vec3 &cameraPos,
        Mat4 &pMat,
        Mat4 &vMat)
    {
        shader.Bind();
        pMat = glm::perspective(1.0472f, aspect, 0.01f, 1000.f);
        shader.SetUnformMat4f("proj_matrix", pMat);
        vMat = glm::translate(Mat4(1.0f), cameraPos * -1.f);
    }

}