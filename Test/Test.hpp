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

    /**
     * @brief Базовый класс для всех Тестов.
     * @details Каждый тест - абстрактный модуль, получающий на вход базовые параметры (положение камеры, frame-rate, соотношение сторон окна а так же матрицу переспективы.)
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