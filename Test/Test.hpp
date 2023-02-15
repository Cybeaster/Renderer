#pragma once

// clang-format off
#include "GL.hpp"
#include "Math.hpp"
#include "Shader.hpp"
#include "SmartPtr.hpp"
#include "Types.hpp"
#include "Vector.hpp"
#include "Vertex/VertexArray.hpp"
#include "Vertex/VertexData/DrawContext.hpp"
#include <Path.hpp>
#include <filesystem>
#include <memory>
#include <stack>

// clang-format on

namespace RenderAPI
{
class ORenderer;
}
namespace Test
{
using RenderAPI::ORenderer;
using RenderAPI::OShader;
using RenderAPI::OTSharedPtr;
using RenderAPI::SDrawContext;
using RenderAPI::SVertexContext;
/**
 * @brief Base class for all tests.
 * @details Each test is an abstract modul, receiving as input base parameters(camera location, frame rate, aspect ration, perspective matrix)
 *
 */
class OTest
{
public:
	OTest(const OPath& shaderPath, const OTSharedPtr<RenderAPI::ORenderer>& RendererArg);
	OTest() = default;
	virtual ~OTest() = default;

	void Init(const OMat4& pMatRef)
	{
		pMat = pMatRef;
	}

	virtual void OnUpdate(
	    const float& DeltaTime,
	    const float& Aspect,
	    const OVec3& CameraPos,
	    OMat4& PMat,
	    OMat4& VMat);

	virtual void OnTestEnd() {}

	virtual void InitShader(const OString& shaderPath);

	void EnableVertexArray(OBuffer& buffer);

	TDrawVertexHandle CreateVertexElement(const SVertexContext& VContext, const SDrawContext& RContext);
	void EnableBuffer(const OBufferAttribVertexHandle& Handle);
	void EnableBuffer(const TDrawVertexHandle& Handle);

	void DrawArrays(const TDrawVertexHandle& Handle);

protected:
	OShader& GetShader()
	{
		return Shader;
	}

	std::stack<OMat4>& GetMVStack()
	{
		return mvStack;
	}

	std::stack<OMat4> mvStack;
	OTVector<GLuint> vertexArray;

	OTSharedPtr<class ORenderer> Renderer;

private:
	OMat4
	    pMat,
	    mMat,
	    mvMat,
	    tMat,
	    rMat;

	/**
	 * @brief Shader that is used for pipeline.
	 *
	 */
	OShader Shader;
	OTVector<TVertexHandle> Handles;
};

} // namespace Test