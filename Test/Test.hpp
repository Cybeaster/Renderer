#pragma once

#include "GL/glew.h"
#include "Math.hpp"
#include "Shader.hpp"
#include "SmartPtr.hpp"
#include "Types.hpp"
#include "Vector.hpp"
#include "Vertex/VertexArray.hpp"
#include "glfw3.h"

#include <Path.hpp>
#include <filesystem>
#include <memory>
#include <stack>

namespace RenderAPI
{
class ORenderer;
}
namespace Test
{
using RenderAPI::ORenderer;
using RenderAPI::OSharedPtr;
using RenderAPI::SVertexContext;
/**
 * @brief Base class for all tests.
 * @details Each test is an abstract modul, receiving as input base parameters(camera location, frame rate, aspect ration, perspective matrix)
 *
 */
class OTest
{
public:
	OTest(const OPath& shaderPath, const OSharedPtr<RenderAPI::ORenderer>& RendererArg);
	OTest() = default;
	virtual ~OTest();

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

	TDrawVertexHandle CreateVertexElement(const SVertexContext& VContext, const TDrawContext& RContext);
	void EnableBuffer(const TBufferAttribVertexHandle& Handle);
	void EnableBuffer(const TDrawVertexHandle& Handle);

	void DrawArrays(const TDrawVertexHandle& Handle);

protected:
	TShader& GetShader()
	{
		return Shader;
	}

	std::stack<OMat4>& GetMVStack()
	{
		return mvStack;
	}

	std::stack<OMat4> mvStack;
	OVector<GLuint> vertexArray;

	OSharedPtr<class ORenderer> Renderer;

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
	TShader Shader;
	OVector<TVertexHandle> Handles;
};

} // namespace Test