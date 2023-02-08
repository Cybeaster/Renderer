#include "TestSimpleCube.hpp"

#include <Renderer.hpp>

namespace Test
{
OTestSimpleCube::OTestSimpleCube(TPath ShaderPath, OSharedPtr<RenderAPI::ORenderer> Renderer)
    : OTest(ShaderPath, Renderer)
{
	TVertexContext contextVertex(
	    new TBuffer{ cubePositions, sizeof(cubePositions) },
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
    const TVec3& cameraPos,
    TMat4& pMat,
    TMat4& vMat)
{
	OTest::OnUpdate(deltaTime, aspect, cameraPos, pMat, vMat);

	mMatrix = glm::translate(TMat4(1), cubePos);
	mvMatrix = mMatrix * vMat;
	GetShader().SetUnformMat4f("mv_matrix", mvMatrix);
	GetShader().SetUniform4f("additionalColor", 1, 1, 1, 1);

	DrawArrays(handle);
	// GLCall(glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0));
	// GLCall(glEnableVertexAttribArray(0));

	// GLCall(glDrawArrays(GL_TRIANGLES, 0, 36));
}
} // namespace Test