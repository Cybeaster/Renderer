//
// Created by Cybea on 5/7/2023.
//

#include "TestLightning.hpp"

#include "Application/Application.hpp"
#include "Assert.hpp"
#include "Materials/Material.hpp"

namespace RAPI
{
OTestLightning::OTestLightning(const OPath& TextureFirstPath, const OPath& SecondTexturePath, const OPath& ShaderPath, RAPI::ORenderer* Renderer)
    : OTest(ShaderPath, Renderer), TorusTexture(TextureFirstPath)
{
	GLCall(glGenVertexArrays(2, VAO));

	GLCall(glGenBuffers(6, VBO));
	GLCall(glGenBuffers(2, EBO));

	Torus.GetVertexTextureNormalPositions(TorusContext);
	auto indices = Torus.GetIndices();

	GLCall(glBindVertexArray(VAO[0]));

	GLCall(glBindBuffer(GL_ARRAY_BUFFER, VBO[0]));
	GLCall(glBufferData(GL_ARRAY_BUFFER, TorusContext.Vertices.size() * sizeof(float), TorusContext.Vertices.data(), GL_STATIC_DRAW));
	GLCall(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr));
	GLCall(glEnableVertexAttribArray(0));

	GLCall(glBindBuffer(GL_ARRAY_BUFFER, VBO[1]));
	GLCall(glBufferData(GL_ARRAY_BUFFER, TorusContext.TexCoords.size() * sizeof(float), TorusContext.TexCoords.data(), GL_STATIC_DRAW));
	GLCall(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr));
	GLCall(glEnableVertexAttribArray(1));

	GLCall(glBindBuffer(GL_ARRAY_BUFFER, VBO[2]));
	GLCall(glBufferData(GL_ARRAY_BUFFER, TorusContext.Normals.size() * sizeof(float), TorusContext.Normals.data(), GL_STATIC_DRAW));
	GLCall(glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr));
	GLCall(glEnableVertexAttribArray(2));

	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[0]));
	GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32), indices.data(), GL_STATIC_DRAW));
	TorusTexture.Bind();

	GLCall(glBindVertexArray(VAO[1]));

	Cube.GetVertexTextureNormalPositions(CubeContext);
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, VBO[3]));
	GLCall(glBufferData(GL_ARRAY_BUFFER, CubeContext.Vertices.size() * sizeof(float), CubeContext.Vertices.data(), GL_STATIC_DRAW));
	GLCall(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr));
	GLCall(glEnableVertexAttribArray(0));

	GLCall(glBindBuffer(GL_ARRAY_BUFFER, VBO[4]));
	GLCall(glBufferData(GL_ARRAY_BUFFER, CubeContext.TexCoords.size() * sizeof(float), CubeContext.TexCoords.data(), GL_STATIC_DRAW));
	GLCall(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr));
	GLCall(glEnableVertexAttribArray(1));

	GLCall(glBindBuffer(GL_ARRAY_BUFFER, VBO[5]));
	GLCall(glBufferData(GL_ARRAY_BUFFER, CubeContext.Normals.size() * sizeof(float), CubeContext.Normals.data(), GL_STATIC_DRAW));
	GLCall(glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr));
	GLCall(glEnableVertexAttribArray(2));

	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[1]));
	GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, Cube.GetIndices().size() * sizeof(uint32), Cube.GetIndices().data(), GL_STATIC_DRAW));
	TorusTexture.Bind();

	GLCall(glBindVertexArray(0));

	MTorusMatrix = glm::rotate(MTorusMatrix, SMath::ToRadians(35.f), { 1.f, 0.0, 0.0 });

	SpotLight.Ambient = { 0, 0, 0, 1.F };
	SpotLight.Specular = { 1.F, 1.F, 1.F, 1.F };
	SpotLight.Diffuse = { 1.F, 1.F, 1.F, 1.F };

	SpotLight.Cutoff = 35.f;
	SpotLight.Position = { 0.0, 5.0, 0.0 };
	SpotLight.Direction = { 0.0, 0.0, 1.F };

	SmallCubeMatrix = glm::scale(glm::translate(SmallCubeMatrix, SpotLight.Position), { 0.1, 0.1, 0.1 });
	CubeMatrix = glm::translate(CubeMatrix, { 0, 0, 3.f });
}

void OTestLightning::InstallLights(OMat4 VMatrix)
{
	auto goldMaterial = OMaterial::GetBronzeMaterial();
	auto multipliedPos = (RayWorld * 3.F) * SpotLight.Position;
	GetShader().SetUniformVec4f("globalAmbient", GlobalAmbient);
	GetShader().SetUniformVec4f("light.base.base.ambient", SpotLight.Ambient);
	GetShader().SetUniformVec4f("light.base.base.diffuse", SpotLight.Diffuse);
	GetShader().SetUniformVec4f("light.base.base.specular", SpotLight.Specular);
	GetShader().SetUniformVec3f("light.base.position", OVec3(VMatrix * OVec4(multipliedPos, 1.0)));

	GetShader().SetUniform1f("light.base.attenuation.constant", 1.F);
	GetShader().SetUniform1f("light.base.attenuation.quadratic", 0.032F);
	GetShader().SetUniform1f("light.base.attenuation.linear", 0.09F);

	GetShader().SetUniformVec4f("material.ambient", goldMaterial.Ambient);
	GetShader().SetUniformVec4f("material.diffuse", goldMaterial.Diffuse);
	GetShader().SetUniformVec4f("material.specular", goldMaterial.Specular);
	GetShader().SetUniform1f("material.shininess", goldMaterial.Shininess);
}

void OTestLightning::OnUpdate(const float& DeltaTime, const float& Aspect, const OVec3& CameraPos, OMat4& PMat, OMat4& VMat)
{
	OTest::OnUpdate(DeltaTime, Aspect, CameraPos, PMat, VMat);
	GetShader().SetUniform1i("useTexture", 0);
	InstallLights(VMat);

	auto ndc = OApplication::GetApplication()->GetWindow()->GetNDC();
	OVec4 rayClip = { ndc.x, ndc.y, -1, 1 };
	auto rayView = glm::inverse(PMat) * rayClip;
	rayView.z = -1;
	rayView.w = 0;
	RayWorld = OVec3(glm::inverse(VMat) * rayView);
	RayWorld = glm::normalize(RayWorld);

	// Calc Torus
	auto MVTorusMatrix = VMat * MTorusMatrix;
	InvTrTorusMat = glm::transpose(glm::inverse(MVTorusMatrix));

	GetShader().SetUniformMat4f("mv_matrix", MVTorusMatrix);
	GetShader().SetUniformMat4f("norm_matrix", InvTrTorusMat);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	GLCall(glBindVertexArray(VAO[0]));
	GLCall(glDrawElements(GL_TRIANGLES, Torus.GetNumIndices(), GL_UNSIGNED_INT, 0));

	MVTorusMatrix = VMat * glm::scale(glm::translate(OMat4(1), { SpotLight.Position.x, SpotLight.Position.y, SpotLight.Position.z }), { 0.5, 0.5, 0.5 });
	GetShader().SetUniformMat4f("mv_matrix", MVTorusMatrix);
	GLCall(glDrawElements(GL_TRIANGLES, Torus.GetNumIndices(), GL_UNSIGNED_INT, 0));

	// Calc Cube
	auto MVCubeMatrix = VMat * CubeMatrix;
	InvTrCubeMat = glm::transpose(glm::inverse(MVCubeMatrix));

	GetShader().SetUniformMat4f("mv_matrix", MVCubeMatrix);
	GetShader().SetUniformMat4f("norm_matrix", InvTrCubeMat);

	GLCall(glBindVertexArray(VAO[1]));
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[1]));
	GLCall(glDrawArrays(GL_TRIANGLES, 0, CubeContext.Vertices.size() / 3));

	// Calc small

	GetShader().SetUniform1f("material.shininess", 100);
	auto MVSecondCubeMatrix = VMat * SmallCubeMatrix;
	auto secondInvCube = glm::transpose(glm::inverse(MVSecondCubeMatrix));

	GetShader().SetUniformMat4f("mv_matrix", MVSecondCubeMatrix);
	GetShader().SetUniformMat4f("norm_matrix", secondInvCube);

	GLCall(glBindVertexArray(VAO[1]));
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[1]));
	GLCall(glDrawArrays(GL_TRIANGLES, 0, CubeContext.Vertices.size() / 3));
}

} // namespace RAPI