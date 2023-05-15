//
// Created by Cybea on 5/14/2023.
//

#include "Plane.hpp"
void RAPI::OPlane::Init(uint32 Precision) noexcept
{
	Vertices = {
		glm::vec3(-1.0f, -1.0f, 0), // 0 Front Bottom Left
		glm::vec3(1.0f, -1.0f, 0), // 1 Front Bottom Right
		glm::vec3(1.0f, 1.0f, 0), // 2 Front Top Right
		glm::vec3(-1.0f, 1.0f, 0), // 3 Front Top Left
	};

	// clang-format off
	Indices = {
		0, 1, 2,
		2, 3, 0  // Front Face
	};

	Normals = {
		glm::vec3(0.0f, 0.0f, -1.0f), // Front face
	};
	// clang-format on

	TexCoords = {
		glm::vec2(0.0f, 0.0f),
		glm::vec2(1.0f, 0.0f),
		glm::vec2(1.0f, 1.0f),
		glm::vec2(0.0f, 1.0f),
	};
	NumIndices = Indices.size();
	NumVertices = Vertices.size();
}
void RAPI::OPlane::PreInit(uint32 Precision) noexcept
{
}
void RAPI::OPlane::GetVertexTextureNormalPositions(SModelContext& OutContext)
{
	for (int i = 0; i < NumIndices; ++i)
	{
		OutContext.Vertices.push_back(Vertices[Indices[i]].x);
		OutContext.Vertices.push_back(Vertices[Indices[i]].y);
		OutContext.Vertices.push_back(Vertices[Indices[i]].z);

		OutContext.TexCoords.push_back(TexCoords[Indices[i]].x);
		OutContext.TexCoords.push_back(TexCoords[Indices[i]].y);

		OutContext.Normals.push_back(Normals[0].x);
		OutContext.Normals.push_back(Normals[0].y);
		OutContext.Normals.push_back(Normals[0].z);
	}
}
