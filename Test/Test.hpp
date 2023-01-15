#pragma once

#include "GL/glew.h"
#include "glfw3.h"

#include "Math.hpp"
#include "Vector.hpp"
#include "Types.hpp"
#include "Shader.hpp"
#include "SmartPtr.hpp"
#include <filesystem>
#include "Vertex/VertexArray.hpp"
#include <stack>
#include <memory>
#include <Path.hpp>

namespace RenderAPI
{
    class TRenderer;
}
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
        Test(TPath shaderPath, TTSharedPtr<RenderAPI::TRenderer> RendererArg);
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

        virtual void InitShader(TString shaderPath);

        void EnableVertexArray(TBuffer &buffer);

        TDrawVertexHandle CreateVertexElement(const TVertexContext &VContext, const TDrawContext &RContext);
        void EnableBuffer(const TBufferAttribVertexHandle &Handle);
        void EnableBuffer(const TDrawVertexHandle &Handle);

        void DrawArrays(const TDrawVertexHandle &Handle);

    protected:
        TShader &GetShader()
        {
            return Shader;
        }

        std::stack<TMat4> &GetMVStack()
        {
            return mvStack;
        }

        std::stack<TMat4> mvStack;
        TTVector<GLuint> vertexArray;

        TTSharedPtr<class TRenderer> Renderer;

    private:
        TMat4
            pMat,
            mMat,
            mvMat,
            tMat,
            rMat;

        /**
         * @brief Shader that is used for pipeline.
         *
         */
        TShader Shader;
        TTVector<TVertexHandle> Handles;
    };

}