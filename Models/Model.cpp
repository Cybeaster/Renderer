//
// Created by Cybea on 2/28/2023.
//

#include "Model.hpp"

namespace RenderAPI
{

void OModel::GetVertexTextureNormalPositions(SModelContext& OutContext)
{
	const auto indices = GetNumIndices();

	for (int idx = 0; idx < indices; ++idx)
	{
		OutContext.VertexCoords.push_back(Vertices[Indices[idx]].x);
		OutContext.VertexCoords.push_back(Vertices[Indices[idx]].y);
		OutContext.VertexCoords.push_back(Vertices[Indices[idx]].z);

		OutContext.TextureCoords.push_back(TexCoords[Indices[idx]].s);
		OutContext.TextureCoords.push_back(TexCoords[Indices[idx]].t);

		OutContext.NormalsCoords.push_back(Normals[Indices[idx]].x);
		OutContext.NormalsCoords.push_back(Normals[Indices[idx]].y);
		OutContext.NormalsCoords.push_back(Normals[Indices[idx]].z);
	}
}
} // namespace RenderAPI