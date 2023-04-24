#pragma once

#include "GeneratedModel.hpp"
namespace RAPI
{

void OGeneratedModel::PreInit(uint32 Precision) noexcept
{
	SqredPrecision = Precision * Precision;
	NumVertices = SqredPrecision + 2 * Precision + 1;
	NumIndices = SqredPrecision * 6;

	Vertices.resize(NumVertices);
	TexCoords.resize(NumVertices);
	Normals.resize(NumVertices);
	Indices.resize(NumIndices);
}
} // namespace RAPI