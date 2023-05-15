//
// Created by Cybea on 5/14/2023.
//

#include "Cube.hpp"
void RAPI::OCube::Init(uint32 Precision) noexcept
{
	Vertices = {
		glm::vec3(-1.0f, -1.0f, 1.0f), // 0 Front Bottom Left
		glm::vec3(1.0f, -1.0f, 1.0f), // 1 Front Bottom Right
		glm::vec3(1.0f, 1.0f, 1.0f), // 2 Front Top Right
		glm::vec3(-1.0f, 1.0f, 1.0f), // 3 Front Top Left
		glm::vec3(-1.0f, -1.0f, -1.0f), // 4 Back Bottom Left
		glm::vec3(1.0f, -1.0f, -1.0f), // 5 Back Bottom Right
		glm::vec3(1.0f, 1.0f, -1.0f), // 6 Back Top Right
		glm::vec3(-1.0f, 1.0f, -1.0f) // 7 Back Top Left

	};

	// clang-format off
	Indices = {
		0, 1, 2, 2, 3, 0, // Front Face
		1, 5, 6, 6, 2, 1, // Right Face
		5, 4, 7, 7, 6, 5, // Back Face
		4, 0, 3, 3, 7, 4, // Left Face
		3, 2, 6, 6, 7, 3, // Top Face
		4, 5, 1, 1, 0, 4  // Bottom Face
	};

	Normals = {
		glm::vec3(0.0f, 0.0f, -1.0f), // Front face
		glm::vec3(0.0f, 0.0f, 1.0f),  // Back face
		glm::vec3(0.0f, -1.0f, 0.0f), // Bottom face
		glm::vec3(0.0f, 1.0f, 0.0f),  // Top face
		glm::vec3(-1.0f, 0.0f, 0.0f), // Left face
		glm::vec3(1.0f, 0.0f, 0.0f)   // Right face
	};
	// clang-format on

	TexCoords = {
		glm::vec2(0.0f, 0.0f),
		glm::vec2(1.0f, 0.0f),
		glm::vec2(1.0f, 1.0f),
		glm::vec2(0.0f, 1.0f),
		glm::vec2(0.0f, 0.0f),
		glm::vec2(1.0f, 0.0f),
		glm::vec2(1.0f, 1.0f),
		glm::vec2(0.0f, 1.0f)
	};
}
void RAPI::OCube::GetVertexTextureNormalPositions(SModelContext& OutContext)
{
	for (int i = 0; i < NumIndices; ++i)
	{
		OutContext.Vertices.push_back(Vertices[Indices[i]].x);
		OutContext.Vertices.push_back(Vertices[Indices[i]].y);
		OutContext.Vertices.push_back(Vertices[Indices[i]].z);

		OutContext.TexCoords.push_back(TexCoords[Indices[i]].x);
		OutContext.TexCoords.push_back(TexCoords[Indices[i]].y);

		OutContext.Normals.push_back(Normals[Indices[i] % 6].x);
		OutContext.Normals.push_back(Normals[Indices[i] % 6].y);
		OutContext.Normals.push_back(Normals[Indices[i] % 6].z);
	}
}
void RAPI::OCube::PreInit(uint32 Precision) noexcept
{
	NumVertices = 24;
	NumIndices = 36;
}
