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
    : OTest(ShaderPath, Renderer), ShadowShader(ShadowShaderPath)
{
	Renderer->SetCameraPosition({ -1.2, 1, 0 });
	AddPointLights();
	AddSpotLights();

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

	SetupShadowBuffers();

	GLCall(glBindVertexArray(0));

	// MTorusMatrix *= glm::rotate(OMat4(1), 45.f, { 0, 1, 0 });
	MTorusMatrix *= glm::translate(OMat4(1), { 0, 1, 0 });
	MTorusMatrix *= glm::scale(OMat4(1), { 0.3, 0.3, 0.3 });

	GlobalAmbient = OVec4(1.f);

	SmallCubeMatrix = glm::scale(SmallCubeMatrix, { 0.1, 0.1, 0.1 });

	DownCubeMat = glm::translate(DownCubeMat, { 0, -2, 0 });
	DownCubeMat = glm::scale(DownCubeMat, { 1, 1, 1 });
}

void OTestLightning::InstallLights(OMat4 VMatrix)
{
	auto goldMaterial = OMaterial::GetGoldMaterial();
	GetShader().SetUniformVec4f("globalAmbient", GlobalAmbient);
	GetShader().SetUniform1ui("numPointLights", PointLights.size());
	GetShader().SetUniform1ui("numSpotLights", SpotLights.size());

	for (int i = 0; i < SpotLights.size(); ++i)
	{
		auto baseName = SLogUtils::Format("spotLights[{}]", i);

		GetShader().SetUniformVec4f(baseName + ".base.base.ambient", SpotLights[i]->Ambient);
		GetShader().SetUniformVec4f(baseName + ".base.base.diffuse", SpotLights[i]->Diffuse);
		GetShader().SetUniformVec4f(baseName + ".base.base.specular", SpotLights[i]->Specular);
		GetShader().SetUniformVec3f(baseName + ".direction", SpotLights[i]->Direction);

		GetShader().SetUniformVec3f(baseName + ".base.position", OVec3(VMatrix * OVec4(SpotLights[i]->Position, 1.0)));

		GetShader().SetUniform1f(baseName + ".base.attenuation.constant", SpotLights[i]->Attenuation.Constant);
		GetShader().SetUniform1f(baseName + ".base.attenuation.quadratic", SpotLights[i]->Attenuation.Quadratic);
		GetShader().SetUniform1f(baseName + ".base.attenuation.linear", SpotLights[i]->Attenuation.Linear);
	}

	for (int i = 0; i < PointLights.size(); ++i)
	{
		auto baseName = SLogUtils::Format("pointLights[{}]", i);

		GetShader().SetUniformVec4f(baseName + ".base.ambient", PointLights[i].Ambient);
		GetShader().SetUniformVec4f(baseName + ".base.diffuse", PointLights[i].Diffuse);
		GetShader().SetUniformVec4f(baseName + ".base.specular", PointLights[i].Specular);

		GetShader().SetUniformVec3f(baseName + ".position", OVec3(VMatrix * OVec4(PointLights[i].Position, 1.0)));

		GetShader().SetUniform1f(baseName + ".attenuation.constant", PointLights[i].Attenuation.Constant);
		GetShader().SetUniform1f(baseName + ".attenuation.quadratic", PointLights[i].Attenuation.Quadratic);
		GetShader().SetUniform1f(baseName + ".attenuation.linear", PointLights[i].Attenuation.Linear);
	}

	GetShader().SetUniformVec4f("material.ambient", goldMaterial.Ambient);
	GetShader().SetUniformVec4f("material.diffuse", goldMaterial.Diffuse);
	GetShader().SetUniformVec4f("material.specular", goldMaterial.Specular);
	GetShader().SetUniform1f("material.shininess", goldMaterial.Shininess);
}

void OTestLightning::SetupShadowBuffers()
{
	GLCall(glClear(GL_DEPTH_BUFFER_BIT));
	GLCall(glClear(GL_COLOR_BUFFER_BIT));

	GLCall(glGenFramebuffers(1, &ShadowBuffer));
	GLCall(glBindFramebuffer(GL_FRAMEBUFFER, ShadowBuffer));

	GLCall(glGenTextures(1, &ShadowTexture));
	GLCall(glBindTexture(GL_TEXTURE_2D, ShadowTexture));

	GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, 1920, 1080, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0));

	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
}

void OTestLightning::OnUpdate(const float& DeltaTime, const float& Aspect, const OVec3& CameraPos, OMat4& PMat, OMat4& VMat)
{
	auto ndc = OApplication::GetApplication()->GetWindow()->GetNDC();
	auto lightPos = PointShadowLight.Position + OVec3(ndc.x, ndc.y, 0.0);

	auto view = glm::lookAt(lightPos, { 0, -1, 0 }, { 0, 1.f, 0 });

	SmallCubeMatrix = glm::scale(glm::translate(OMat4(1), lightPos + OVec3(0, 0.2, 0)), { 0.1, 0.1, 0.1 });

	LightViewMat = view;
	LightPMat = PMat;

	CurrentAngle += DeltaTime;
	if (CurrentAngle > 360)
		CurrentAngle = 0;
	MTorusMatrix = glm::rotate(glm::translate(OMat4(1), { cos(CurrentAngle), 0, sin(CurrentAngle) }), CurrentAngle, { 1, 0.5, 0 });

	auto size = OApplication::GetApplication()->GetWindow()->GetWidthHeight();
	GetShader().SetUniform1f("windowHeight", size.y);
	GetShader().SetUniform1f("windowWidth", size.x);

	GLCall(glBindVertexArray(0));
	GLCall(glBindFramebuffer(GL_FRAMEBUFFER, ShadowBuffer));
	GLCall(glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, ShadowTexture, 0));
	ASSERT(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

	GLCall(glDrawBuffer(GL_NONE));
	GLCall(glEnable(GL_DEPTH_TEST));

	ShadowShader.Bind();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glCullFace(GL_BACK);

	// glEnable(GL_POLYGON_OFFSET_FILL);

	// glPolygonOffset(2.0f, 4.0f);

	GLCall(glBindBuffer(GL_ARRAY_BUFFER, VBO[3]));
	GLCall(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0));
	GLCall(glEnableVertexAttribArray(0));

	ShadowShader.SetUniformMat4f("shadowMVP", LightPMat * LightViewMat * SmallCubeMatrix);
	GLCall(glDrawArrays(GL_TRIANGLES, 0, CubeContext.Vertices.size() / 3));

	ShadowShader.SetUniformMat4f("shadowMVP", LightPMat * LightViewMat * DownCubeMat);
	GLCall(glDrawArrays(GL_TRIANGLES, 0, CubeContext.Vertices.size() / 3));

	GLCall(glBindBuffer(GL_ARRAY_BUFFER, VBO[0]));
	GLCall(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0));
	GLCall(glEnableVertexAttribArray(0));

	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[0]));
	ShadowShader.SetUniformMat4f("shadowMVP", LightPMat * LightViewMat * MTorusMatrix);
	GLCall(glDrawElements(GL_TRIANGLES, Torus.GetNumIndices(), GL_UNSIGNED_INT, nullptr));

	// glDisable(GL_POLYGON_OFFSET_FILL);

	GLCall(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	GLCall(glActiveTexture(GL_TEXTURE0));
	GLCall(glBindTexture(GL_TEXTURE_2D, ShadowTexture));
	GLCall(glDrawBuffer(GL_FRONT));

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glCullFace(GL_BACK);

	OTest::OnUpdate(DeltaTime, Aspect, CameraPos, PMat, VMat);
	ComputeLight(VMat, CameraPos, PMat);

	DrawModelVertices(VAO[1], DownCubeMat, VMat, CubeContext);
	DrawModelVertices(VAO[1], SmallCubeMatrix, VMat, CubeContext);
	DrawModelIndices(VAO[0], MTorusMatrix, VMat, Torus.GetNumIndices());
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
	GetShader().SetUniformMat4f("shadowMVP", BiasesMat * LightPMat * LightViewMat * ModelMatrix);
}

void OTestLightning::ComputeLight(const OMat4& VMat, const OVec3& CameraPos, const OMat4& PMat)
{
	InstallLights(VMat);
}

void OTestLightning::ComputeShadows()
{
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

void OTestLightning::AddPointLights()
{
	auto mainSpot = MakeShared<SSpotlight>();

	mainSpot->Ambient = { 0, 0, 0, 1.F };
	mainSpot->Specular = { 1.F, 1.F, 1.F, 1.F };
	mainSpot->Diffuse = { 1.F, 1.F, 1.F, 1.F };
	mainSpot->Position = OVec3{ 0, 2, 0 };
	mainSpot->Direction = { 0.0, -1, -1.f };
	mainSpot->Cutoff = 75.f;
	mainSpot->Attenuation = DefaultAttenuation;

	// SpotLights.push_back(Move(mainSpot));
}

void OTestLightning::AddSpotLights()
{
	PointShadowLight.Position = { -0.5, 1, 0 };
	PointShadowLight.Specular = { 1.F, 1.F, 1.F, 1.F };
	PointShadowLight.Diffuse = { 1.F, 1.F, 1.F, 1.F };
	PointShadowLight.Ambient = { 0, 0, 0, 1.F };
	PointShadowLight.Attenuation = DefaultAttenuation;

	PointLights.push_back(PointShadowLight);
}

} // namespace RAPI