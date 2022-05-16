#include "Test.hpp"
#include <Renderer.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
namespace test
{

    Test::Test(std::string shaderPath) : shader(shaderPath)
    {
        
    }

    Test::~Test()
    {

    }
    void Test::AddVertexArray()
    {
        vertexArray.push_back({});
        GLuint* vaID = &vertexArray[vertexArray.size() - 1];

        GLCall(glGenVertexArrays(1,vaID));
        GLCall(glBindVertexArray(*vaID));
    }


    void Test::AddBuffers(std::vector<std::vector<float>>& vertecis,size_t numOfBuffers)
    {
        for (size_t i = 0; i < numOfBuffers; i++)
        {
           
        }
    }

    void Test::InitShader(std::string shaderPath)
    {
        shader.Init(shaderPath);
    }

    void Test::EnableVertexArray(GLuint bufferID)
    {
        buffers[bufferID]->Bind();
        GLCall(glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,0));
        GLCall(glEnableVertexAttribArray(0));
    }

    void Test::OnUpdate(
        GLFWwindow* window,
         const float deltaTime,
         const float aspect,
         const glm::vec3& cameraPos,
         glm::mat4& pMat,
         glm::mat4& vMat)
    {
       shader.Bind();
        
        pMat = glm::perspective(1.0472f,aspect,0.01f,1000.f);
        shader.SetUnformMat4f("proj_matrix",pMat);
        vMat = glm::translate(glm::mat4(1.0f),cameraPos * -1.f);

    }
    void Test::AddBuffer(void* buffer,int32_t size)
    {
        buffers.push_back(std::make_shared<VertexBuffer>(buffer,size));
    }
}