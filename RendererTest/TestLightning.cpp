//
// Created by Cybea on 5/7/2023.
//

#include "TestLightning.hpp"

#include "Application/Application.hpp"
#include "Assert.hpp"
#include "Materials/Material.hpp"

namespace RAPI
{
OTestLightning::OTestLightning(const OPath& TextureFirstPath, const OPath& SecondTexturePath, const OPath& ShaderPath, RAPI::ORenderer* Renderer, const OPath& ShadowShaderPath)
    : OTest(ShaderPath, Renderer), TorusTexture(TextureFirstPath), ShadowShader(ShadowShaderPath)
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

	SetupShadowBuffers();

	MTorusMatrix *= glm::translate(OMat4(1), { 0, 0.5, 0 });
	MTorusMatrix *= glm::rotate(OMat4(1), SMath::ToRadians(45.f), { 0, 1.0, 0.0 });

	GlobalAmbient = OVec4(1.f);
	SpotLight.Ambient = { 0, 0, 0, 1.F };
	SpotLight.Specular = { 1.F, 1.F, 1.F, 1.F };
	SpotLight.Diffuse = { 1.F, 1.F, 1.F, 1.F };
	SpotLight.Position = OVec3{ 0, 3, 0 };
	SpotLight.Direction = { 0.0, -1.f, 0 };
	SpotLight.Cutoff = 75.f;

	SmallCubeMatrix = glm::scale(glm::translate(SmallCubeMatrix, SpotLight.Position), { 0.1, 0.1, 0.1 });

	CubeMatrix = glm::translate(CubeMatrix, { 0, 0, 0 });
	CubeMatrix = glm::scale(CubeMatrix, { 1, 1, 1 });

	DownCubeMat = glm::translate(DownCubeMat, { 0, -5, 0 });
	DownCubeMat = glm::scale(DownCubeMat, { 10, 1, 10 });
}

void OTestLightning::InstallLights(OMat4 VMatrix)
{
	auto goldMaterial = OMaterial::GetGoldMaterial();
	GetShader().Bind();
	GetShader().SetUniformVec4f("globalAmbient", GlobalAmbient);
	GetShader().SetUniformVec4f("light.base.base.ambient", SpotLight.Ambient);
	GetShader().SetUniformVec4f("light.base.base.diffuse", SpotLight.Diffuse);
	GetShader().SetUniformVec4f("light.base.base.specular", SpotLight.Specular);
	GetShader().SetUniformVec3f("light.direction", SpotLight.Direction);

	GetShader().SetUniformVec3f("light.base.position", OVec3(VMatrix * OVec4(SpotLight.Position, 1.0)));

	GetShader().SetUniform1f("light.base.attenuation.constant", 1.F);
	GetShader().SetUniform1f("light.base.attenuation.quadratic", 0.032F);
	GetShader().SetUniform1f("light.base.attenuation.linear", 0.09F);

	GetShader().SetUniformVec4f("material.ambient", goldMaterial.Ambient);
	GetShader().SetUniformVec4f("material.diffuse", goldMaterial.Diffuse);
	GetShader().SetUniformVec4f("material.specular", goldMaterial.Specular);
	GetShader().SetUniform1f("material.shininess", goldMaterial.Shininess);
}

void OTestLightning::SetupShadowBuffers()
{
	auto WH = OApplication::GetApplication()->GetWindow()->GetWidthHeight();

	GLCall(glGenFramebuffers(1, &ShadowBuffer));

	GLCall(glGenTextures(1, &ShadowTexture));
	GLCall(glBindTexture(GL_TEXTURE_2D, ShadowTexture));
	GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, WH.x, WH.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0));

	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL));
}

void OTestLightning::OnUpdate(const float& DeltaTime, const float& Aspect, const OVec3& CameraPos, OMat4& PMat, OMat4& VMat)
{
	ComputeShadows(Aspect, SpotLight.Direction, SpotLight.Position);

	GLCall(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	GLCall(glActiveTexture(GL_TEXTURE0));
	GLCall(glBindTexture(GL_TEXTURE_2D, ShadowTexture));
	GLCall(glDrawBuffer(GL_FRONT));

	OTest::OnUpdate(DeltaTime, Aspect, CameraPos, PMat, VMat);
	GetShader().SetUniform1i("useTexture", 1);
	ComputeLight(VMat, CameraPos, PMat);

	DrawModelIndices(VAO[0], VMat, MTorusMatrix, Torus.GetNumIndices());
	DrawModelVertices(VAO[1], DownCubeMat, VMat, CubeContext);
}

void OTestLightning::DrawModelVertices(uint32 VAOIdx, const OMat4& ModelMatrix, const OMat4& VMat, const SModelContext& Context)
{
	SetupNormalAndMVMatrices(VMat, ModelMatrix);
	GLCall(glBindVertexArray(VAOIdx));
	GLCall(glDrawArrays(GL_TRIANGLES, 0, Context.Vertices.size() / 3));
}

void OTestLightning::DrawModelIndices(uint32 VAOIdx, const OMat4& ModelMatrix, const OMat4& VMat, uint32 NumIndices)
{
	SetupNormalAndMVMatrices(VMat, ModelMatrix);
	GLCall(glBindVertexArray(VAOIdx));
	GLCall(glDrawElements(GL_TRIANGLES, NumIndices, GL_UNSIGNED_INT, nullptr));
}

void OTestLightning::SetupNormalAndMVMatrices(const OMat4& VMat, const OMat4& ModelMatrix)
{
	GetShader().Bind();

	auto MVMatrix = VMat * ModelMatrix;
	auto InvTrMatrix = glm::transpose(glm::inverse(MVMatrix));

	GetShader().SetUniformMat4f("mvMatrix", MVMatrix);
	GetShader().SetUniformMat4f("normMatrix", InvTrMatrix);
	GetShader().SetUniformMat4f("shadowMVP", BiasesMat * SpotLightPVMat * ModelMatrix);
}

void OTestLightning::ComputeLight(const OMat4& VMat, const OVec3& CameraPos, const OMat4& PMat)
{
	InstallLights(VMat);
}

void OTestLightning::ComputeShadows(float Aspect, OVec3& LightDir, OVec3& LightPos)
{
	ShadowShader.Bind();

	auto rightVec = glm::cross({ 0, 1, 0 }, LightDir);
	auto upVec = glm::cross(LightDir, rightVec);
	auto view = glm::lookAt(LightPos, LightPos + LightDir, upVec);
	auto pMat = glm::perspective(SMath::ToRadians(60.f), Aspect, 0.1f, 1000.f);
	SpotLightPVMat = pMat * view;

	GLCall(glBindFramebuffer(GL_FRAMEBUFFER, ShadowBuffer));
	GLCall(glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, ShadowTexture, 0));

	GLCall(glDrawBuffer(GL_NONE));
	GLCall(glEnable(GL_DEPTH_TEST));
	GLCall(glClear(GL_DEPTH_BUFFER_BIT));

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	GLCall(glBindBuffer(GL_ARRAY_BUFFER, VBO[0]));
	GLCall(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0));
	GLCall(glEnableVertexAttribArray(0));

	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[0]));
	ShadowShader.SetUniformMat4f("shadowMVP", SpotLightPVMat * MTorusMatrix);
	GLCall(glDrawElements(GL_TRIANGLES, Torus.GetNumIndices(), GL_UNSIGNED_INT, nullptr));

	GLCall(glBindBuffer(GL_ARRAY_BUFFER, VBO[3]));
	GLCall(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0));
	GLCall(glEnableVertexAttribArray(0));

	ShadowShader.SetUniformMat4f("shadowMVP", SpotLightPVMat * DownCubeMat);
	GLCall(glDrawArrays(GL_TRIANGLES, 0, CubeContext.Vertices.size()));
}

OVec3 OTestLightning::ComputeRayView(const OMat4& PMat, const OMat4& VMat)
{
	auto ndc = OApplication::GetApplication()->GetWindow()->GetNDC();
	OVec4 rayClip = { ndc.x, ndc.y, -1, 1 };
	auto rayView = glm::inverse(PMat) * rayClip;
	rayView.z = -1;
	rayView.w = 0;
	RayWorld = OVec3(glm::inverse(VMat) * rayView);
	RayWorld = glm::normalize(RayWorld);
	return RayWorld;
}

void OTestLightning::CalcModelMatrices()
{
}

} // namespace RAPI