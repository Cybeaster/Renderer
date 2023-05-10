//
// Created by Cybea on 5/7/2023.
//

#include "TestLightning.hpp"

#include "Assert.hpp"
#include "Materials/Material.hpp"

namespace RAPI
{
OTestLightning::OTestLightning(const OPath& TextureFirstPath, const OPath& SecondTexturePath, const OPath& ShaderPath, RAPI::ORenderer* Renderer)
    : OTest(ShaderPath, Renderer), TorusTexture(TextureFirstPath)
{
	GLCall(glGenVertexArrays(1, VAO));
	GLCall(glBindVertexArray(VAO[0]));

	GLCall(glGenBuffers(4, VBO));
	Torus.GetVertexTextureNormalPositions(TorusContext);
	auto indices = Torus.GetIndices();

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

	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO[3]));
	GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(int32), indices.data(), GL_STATIC_DRAW));

	GLCall(glBindVertexArray(0));

	MTorusMatrix = glm::rotate(MTorusMatrix, SMath::ToRadians(35.f), { 1.f, 0.0, 0.0 });
}

void OTestLightning::InstallLights(OMat4 VMatrix)
{
	LightPosV = OVec3(VMatrix * CurrentLightPos);
	LightPos[0] = LightPosV.x;
	LightPos[1] = LightPosV.y;
	LightPos[2] = LightPosV.z;

	auto goldMaterial = OMaterial::GetSilverMaterial();
	GetShader().SetUniformVec4f("globalAmbient", GlobalAmbient);
	GetShader().SetUniformVec4f("light.ambient", LightAmbient);
	GetShader().SetUniformVec4f("light.diffuse", LightDiffuse);
	GetShader().SetUniformVec4f("light.specular", LightSpecular);
	GetShader().SetUniformVec3f("light.position", LightPos);
	GetShader().SetUniformVec4f("material.ambient", goldMaterial.Ambient);
	GetShader().SetUniformVec4f("material.diffuse", goldMaterial.Diffuse);
	GetShader().SetUniformVec4f("material.specular", goldMaterial.Specular);
	GetShader().SetUniform1f("material.shininess", goldMaterial.Shininess);
}

void OTestLightning::OnUpdate(const float& DeltaTime, const float& Aspect, const OVec3& CameraPos, OMat4& PMat, OMat4& VMat)
{
	OTest::OnUpdate(DeltaTime, Aspect, CameraPos, PMat, VMat);

	GetShader().SetUniform1i("use_texture", 0);
	TorusTexture.Bind();

	InstallLights(VMat);

	auto MVMatrix = VMat * MTorusMatrix;
	InvTrMat = glm::transpose(glm::inverse(MVMatrix));

	CurrentLightPos = { InitialLightLoc.x, InitialLightLoc.y, InitialLightLoc.z, 1.0 };
	CurrentLightPos *= OVec4(OVec3(cos(DeltaTime)), 1.0);
	GetShader().SetUniformMat4f("mv_matrix", MVMatrix);
	GetShader().SetUniformMat4f("norm_matrix", InvTrMat);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	GLCall(glBindVertexArray(VAO[0]));
	GLCall(glDrawElements(GL_TRIANGLES, Torus.GetNumIndices(), GL_UNSIGNED_INT, 0));
}
} // namespace RAPI