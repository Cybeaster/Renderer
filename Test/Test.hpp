#pragma once
#include "Math.hpp"
#include "Vector.hpp"
#include "Types.hpp"
#include "GL/glew.h"
#include "glfw3.h"
#include "Shader.hpp"
#include <vector>

#include "Vertex/VertexArrayElem.hpp"
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
        Test(TString shaderPath);
        Test() = default;
        virtual ~Test();

        void Init(const TMat4 &pMatRef)
        {
            pMat = pMatRef;
        }

        virtual void OnUpdate(
            const float deltaTime,
            const float aspect,
            const TVec3 &cameraPos,
            TMat4 &pMat,
            TMat4 &vMat);

        virtual void OnTestEnd() {}

        void AddBuffers(TTVector<TTVector<float>> &vertecis, size_t numOfBuffers);
        void AddBuffer(void *buffer, int32_t size);

        virtual void InitShader(TString shaderPath);
        virtual void EnableVertexArray(GLuint bufferID);
        void EnableVertexArray(TBuffer &buffer);

    protected:
        void AddVertexArray();

        Shader &getShader()
        {
            return shader;
        }

        std::stack<TMat4> &GetMVStack()
        {
            return mvStack;
        }

        std::stack<TMat4> mvStack;
        TTVector<GLuint> vertexArray;
        TTVector<std::shared_ptr<TBuffer>> buffers;

    private:
        TMat4
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