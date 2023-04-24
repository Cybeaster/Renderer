//
// Created by Cybea on 4/16/2023.
//
#include "TestModelling.hpp"

#include "Assert.hpp"

namespace RAPI
{

OTestModelling::OTestModelling(const OPath& ShaderPath, ORenderer* Renderer, RAPI::OModel* TestModel, const OPath& ModelTexturePath)
    : OTest(ShaderPath, Renderer), ModelTexture(ModelTexturePath)
{
	Model = TestModel;
	Model->GetVertexTextureNormalPositions(ModelContext);

	ModelHandle = AddAttribBuffer(
	    SVertexContext(AddBuffer(ModelContext.Vertices.data(), ModelContext.Vertices.size() * 4), 0, 3, GL_FLOAT, false, 0, 0, nullptr));

	TextureHandle = CreateVertexElement(
	    SVertexContext(AddBuffer(ModelContext.TexCoords.data(), ModelContext.TexCoords.size() * 4), 1, 2, GL_FLOAT, false, 1, 0, nullptr),
	    SDrawContext(GL_TRIANGLES, 0, ModelContext.TexCoords.size(), GL_LEQUAL, GL_CCW, GL_DEPTH_TEST));

	//	GLCall(glBindBuffer(GL_ARRAY_BUFFER, VBO[1]));
	//	GLCall(glBufferData(GL_ARRAY_BUFFER, context.TexCoords.size() * 4, &context.TexCoords[0], GL_STATIC_DRAW));

	//	GLCall(glBindBuffer(GL_ARRAY_BUFFER, VBO[2]));
	// GLCall(glBufferData(GL_ARRAY_BUFFER, context.Normals.size() * 4, &context.Normals[0], GL_STATIC_DRAW));
}

void OTestModelling::OnUpdate(const float& deltaTime, const float& aspect, const OVec3& cameraPos, OMat4& pMat, OMat4& VMat)
{
	OTest::OnUpdate(deltaTime, aspect, cameraPos, pMat, VMat);

	ModelTexture.Bind(0);

	GetShader().SetUnformMat4f("mv_matrix", glm::scale(glm::mat4(1.F), { 0.2F, 0.2F, 0.2F }) * glm::translate(VMat, { 0, 2, 0 }));

	EnableAttribArrayBuffer(ModelHandle);
	EnableAttribArrayBuffer(TextureHandle);

	GLCall(glDrawArrays(GL_TRIANGLES,
	                    0,
	                    ModelContext.Vertices.size() / 3));
}

} // namespace RAPI