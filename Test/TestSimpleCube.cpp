#include "TestSimpleCube.hpp"

#include <Renderer.hpp>

namespace Test
{
OTestSimpleCube::OTestSimpleCube(OPath ShaderPath, OSharedPtr<RenderAPI::ORenderer> Renderer)
    : OTest(ShaderPath, Renderer)
{
	SVertexContext contextVertex(
	    AddBuffer(cubePositions, sizeof(cubePositions)),
	    0,
	    3,
	    GL_FLOAT,
	    false,
	    0,
	    0,
	    nullptr);

	SDrawContext drawContext(GL_TRIANGLES,
	                         0,
	                         108 / 3,
	                         GL_LEQUAL,
	                         GL_CCW,
	                         GL_DEPTH_TEST);

	handle = CreateVertexElement(contextVertex, drawContext);
}

void OTestSimpleCube::OnUpdate(
    const float& DeltaTime,
    const float& Aspect,
    const OVec3& CameraPos,
    OMat4& PMat,
    OMat4& VMat)
{
	OTest::OnUpdate(DeltaTime, Aspect, CameraPos, PMat, VMat);

	mMatrix = glm::translate(OMat4(1), cubePos);
	mvMatrix = mMatrix * VMat;
	GetShader().SetUnformMat4f("mv_matrix", mvMatrix);
	GetShader().SetUniform4f("additionalColor", 1, 1, 1, 1);

	Draw(handle);
	// GLCall(glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0));
	// GLCall(glEnableVertexAttribArray(0));

	// GLCall(glDrawArrays(GL_TRIANGLES, 0, 36));
}
} // namespace Test