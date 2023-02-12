
#include "Math.hpp"

#include <TestSimpleSolarSystem.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>


namespace Test
{

void OTestSimpleSolarSystem::OnUpdate(
    const float& DeltaTime,
    const float& Aspect,
    const OVec3& CameraPos,
    OMat4& PMat,
    OMat4& VMat)
{
	OTest::OnUpdate(DeltaTime, Aspect, CameraPos, PMat, VMat);

	GetMVStack().push(VMat);
	GetMVStack().push(GetMVStack().top());

	GetMVStack().top() *= glm::translate(OMat4(1.0f), OVec3(0.0, 0.0, 0.0));
	GetMVStack().push(GetMVStack().top());

	GetMVStack().top() *= glm::rotate(OMat4(1.0f), float(DeltaTime), OVec3(1.0, 0.0, 0.0));
	GetShader().SetUnformMat4f("mv_matrix", GetMVStack().top());

	DrawArrays(pyramidHandle);

	GetMVStack().pop();
	// // pyr

	// // cube

	// GetMVStack().push(GetMVStack().top());
	// GetMVStack().top() *= glm::translate(OMat4(1.0f), OVec3(sin(float(DeltaTime)) * 4.0f, 0.0f, cos(float(DeltaTime) * 4.0)));
	// GetMVStack().push(GetMVStack().top());
	// GetMVStack().top() *= glm::rotate(OMat4(1.0f), float(DeltaTime), OVec3(0.0, 1.0, 0.0));
	// GetShader().SetUnformMat4f("mv_matrix", GetMVStack().top());

	// DrawArrays(cubeHandle);

	// GetMVStack().pop();

	// // smaller cube

	// GetMVStack().push(GetMVStack().top());
	// GetMVStack().top() *= glm::translate(OMat4(1.0f), OVec3(0.f, sin(DeltaTime) * 2.0, cos(float(DeltaTime) * 2.0)));
	// GetMVStack().top() *= glm::rotate(OMat4(1.0f), float(DeltaTime), OVec3(0.0, 0.0, 1.0));
	// GetMVStack().top() *= glm::scale(OMat4(1.0), OVec3(0.25f, 0.25f, 0.25f));
	// GetShader().SetUnformMat4f("mv_matrix", GetMVStack().top());

	// DrawArrays(cubeHandle);

	//   GetMVStack().pop();
	// GetMVStack().pop();
	// GetMVStack().pop();
	// GetMVStack().pop();
}

OTestSimpleSolarSystem::OTestSimpleSolarSystem(OPath shaderPath, OTSharedPtr<RenderAPI::ORenderer> Renderer)
    : OTest(shaderPath, Renderer)
{
	// auto size = sizeof(cubePositions);
	// auto data = cubePositions;
	// SVertexContext contextVertex(new OBuffer{data, size}, 0, 3, GL_FLOAT, false, 0, 0, nullptr);

	// SDrawContext drawContext(GL_TRIANGLES,
	//                          0,
	//                          size / 3,
	//                          GL_LEQUAL,
	//                          GL_CCW,
	//                          GL_DEPTH_TEST);
	// cubeHandle = CreateVertexElement(contextVertex, drawContext);

	auto size = sizeof(pyramidPositions);
	auto data = pyramidPositions;

	SVertexContext pyramidVertex(new OBuffer{ pyramidPositions, sizeof(pyramidPositions) }, 0, 3, GL_FLOAT, false, 0, 0, nullptr);

	SDrawContext pyramidDrawContext(GL_TRIANGLES,
	                                0,
	                                54 / 3,
	                                GL_LEQUAL,
	                                GL_CCW,
	                                GL_DEPTH_TEST);

	pyramidHandle = CreateVertexElement(pyramidVertex, pyramidDrawContext);
}

} // namespace Test
