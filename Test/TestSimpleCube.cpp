#include "TestSimpleCube.hpp"

#include <Renderer.hpp>

namespace Test
{
OTestSimpleCube::OTestSimpleCube(OPath ShaderPath, OSharedPtr<RenderAPI::ORenderer> Renderer)
    : OTest(ShaderPath, Renderer)
{
	SVertexContext contextVertex(
	    new OBuffer{ cubePositions, sizeof(cubePositions) },
	    0,
	    3,
	    GL_FLOAT,
	    false,
	    0,
	    0,
	    nullptr);

	TDrawContext drawContext(GL_TRIANGLES,
	                         0,
	                         108 / 3,
	                         GL_LEQUAL,
	                         GL_CCW,
	                         GL_DEPTH_TEST);

	handle = CreateVertexElement(contextVertex, drawContext);
}

void OTestSimpleCube::OnUpdate(
    const float deltaTime,
    const float aspect,
    const OVec3& cameraPos,
    OMat4& pMat,
    OMat4& vMat)
{
	OTest::OnUpdate(deltaTime, aspect, cameraPos, pMat, vMat);

	mMatrix = glm::translate(OMat4(1), cubePos);
	mvMatrix = mMatrix * vMat;
	GetShader().SetUnformMat4f("mv_matrix", mvMatrix);
	GetShader().SetUniform4f("additionalColor", 1, 1, 1, 1);

	DrawArrays(handle);
	// GLCall(glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0));
	// GLCall(glEnableVertexAttribArray(0));

	// GLCall(glDrawArrays(GL_TRIANGLES, 0, 36));
}
} // namespace Test