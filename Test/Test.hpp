#pragma once

#include <RenderAPI.hpp>
#include "GL/glew.h"
#include "glfw3.h"
#include "Shader.hpp"
#include <vector>

#include "VertexBuffer.hpp"
#include <stack>
#include <memory>

namespace Test
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
        Test(String shaderPath);
        Test() = default;
        virtual ~Test();

        void Init(const Mat4 &pMatRef)
        {
            pMat = pMatRef;
        }

        virtual void OnUpdate(
            const float deltaTime,
            const float aspect,
            const Vec3 &cameraPos,
            Mat4 &pMat,
            Mat4 &vMat);

        virtual void OnTestEnd() {}

        void AddBuffers(Vector<Vector<float>> &vertecis, size_t numOfBuffers);
        void AddBuffer(void *buffer, int32_t size);

        virtual void InitShader(String shaderPath);
        virtual void EnableVertexArray(GLuint bufferID);

    protected:
        void AddVertexArray();

        Shader &getShader()
        {
            return shader;
        }

        std::stack<Mat4> &GetMVStack()
        {
            return mvStack;
        }

        std::stack<Mat4> mvStack;
        Vector<GLuint> vertexArray;
        Vector<std::shared_ptr<VertexBuffer>> buffers;

    private:
        Mat4
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