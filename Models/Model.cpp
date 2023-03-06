//
// Created by Cybea on 2/28/2023.
//

#include "Model.hpp"

namespace RenderAPI
{

void OModel::PreInit(int32 Precision) noexcept
{
	SqredPrecision = Precision * Precision;
	NumVertices = SqredPrecision + 2 * Precision + 1;
	NumIndices = SqredPrecision * 6;

	Vertices.resize(NumVertices);
	TexCoords.resize(NumVertices);
	Normals.resize(NumVertices);
	Indices.resize(NumIndices);
}
} // namespace RenderAPI