#pragma once
#include "GL/glew.h"
#include "glfw3.h"
#include "Shader.hpp"
#include <vector>

#include "VertexBuffer.hpp"
#include <stack>
#include <memory>

namespace test
{
    using namespace RenderAPI;

    /**
     * @brief Base class for all tests.
     * @details Each test is an abstract modul, receiving as input base parameters(camera location, frame rate, aspect ration, perspective matrix)
     * 
     */
    class Test
    {

    public:

        Test(std::string shaderPath);
        Test() = default;
        virtual ~Test();
        
        
        void Init(const glm::mat4& pMatRef)
        {pMat = pMatRef;}

        virtual void OnUpdate(
         const float deltaTime,
         const float aspect,
         const glm::vec3& cameraPos,
         glm::mat4& pMat,
         glm::mat4& vMat);
        virtual void OnTestEnd(){}
        void AddBuffers(std::vector<std::vector<float>>& vertecis,size_t numOfBuffers);
        void AddBuffer(void* buffer,int32_t size);

        virtual void InitShader(std::string shaderPath);
        virtual void EnableVertexArray(GLuint bufferID);
        
    protected:
        void AddVertexArray();

        Shader& getShader()
        {return shader;}

        std::stack<glm::mat4>& GetMVStack()
        {return mvStack;}


        std::stack<glm::mat4> mvStack;
        std::vector<GLuint> vertexArray;
        std::vector<std::shared_ptr<VertexBuffer>> buffers;
    private:

        glm::mat4
        pMat,
        mMat,
        mvMat,
        tMat,
        rMat;
        
        /**
         * @brief Шейдер, в который будут отправляться данные по пайплайну.
         * 
         */
        Shader shader;

   

    };
    
 
}