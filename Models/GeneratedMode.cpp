#pragma once

#include "GeneratedModel.hpp"
namespace RAPI
{

void OGeneratedModel::PreInit(uint32 Precision) noexcept
{
	SqredPrecision = Precision * Precision;
	NumVertices = (Precision + 1) * (Precision + 1);
	NumIndices = SqredPrecision * 6;

	for (int i = 0; i < NumVertices; i++)
	{
		Vertices.emplace_back();
	}
	for (int i = 0; i < NumVertices; i++)
	{
		TexCoords.emplace_back();
	}
	for (int i = 0; i < NumVertices; i++)
	{
		Normals.emplace_back();
	}

	for (int i = 0; i < NumIndices; i++)
	{
		Indices.push_back(0);
	}
}
} // namespace RAPI