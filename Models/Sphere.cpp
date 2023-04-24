
#include "Sphere.hpp"

namespace RAPI
{

void OSphere::Init(uint32 Precision) noexcept
{
	// Because the whole sphere has to be "closed";
	const auto incPrecision = Precision + 1;

	for (size_t sliceIndex = 0; sliceIndex <= Precision; ++sliceIndex)
	{
		for (int vertexIndex = 0; vertexIndex < Precision; ++vertexIndex)
		{
			float y = (cos(SMath::ToRadians(180.F - sliceIndex * 180.F / Precision))); // NOLINT
			float x = -(cos(SMath::ToRadians(vertexIndex * 360.F / Precision))) * abs(cos(asin(y))); // NOLINT
			float z = (sin(SMath::ToRadians(vertexIndex * 360.F / Precision))) * abs(cos(asin(y))); // NOLINT

			Vertices[sliceIndex * (incPrecision) + vertexIndex] = OVec3(x, y, z);
			TexCoords[sliceIndex * (incPrecision) + vertexIndex] = OVec2(((float)vertexIndex / Precision), ((float)sliceIndex / Precision)); // NOLINT
			Normals[sliceIndex * (incPrecision) + vertexIndex] = OVec3(x, y, z);
		}
	}

	// Calculate triangles indices
	for (int sliceIdx = 0; sliceIdx < Precision; ++sliceIdx)
	{
		for (int vertexIdx = 0; vertexIdx < Precision; ++vertexIdx)
		{
			Indices[6 * (sliceIdx * Precision + vertexIdx) + 0] = sliceIdx * incPrecision + vertexIdx; // 1t index is the left down corner
			Indices[6 * (sliceIdx * Precision + vertexIdx) + 1] = sliceIdx * incPrecision + vertexIdx + 1; // 2nd and 4th are the same

			// next slice
			Indices[6 * (sliceIdx * Precision + vertexIdx) + 2] = (sliceIdx + 1) * incPrecision + vertexIdx; // 3rd and 6th are the same

			Indices[6 * (sliceIdx * Precision + vertexIdx) + 3] = sliceIdx * incPrecision + vertexIdx + 1; // 2nd and 4th are the same

			// next slice
			Indices[6 * (sliceIdx * Precision + vertexIdx) + 4] = (sliceIdx + 1) * incPrecision + vertexIdx + 1; // 5th index is the up right corner
			Indices[6 * (sliceIdx * Precision + vertexIdx) + 5] = (sliceIdx + 1) * incPrecision + vertexIdx; // 3rd and 6th are the same
		}
	}
}
void OSphere::GetVertexTextureNormalPositions(SModelContext& OutContext)
{
	const auto indices = GetNumIndices();

	for (int idx = 0; idx < indices; ++idx)
	{
		OutContext.Vertices.push_back(Vertices[Indices[idx]].x);
		OutContext.Vertices.push_back(Vertices[Indices[idx]].y);
		OutContext.Vertices.push_back(Vertices[Indices[idx]].z);

		OutContext.TexCoords.push_back(TexCoords[Indices[idx]].s);
		OutContext.TexCoords.push_back(TexCoords[Indices[idx]].t);

		OutContext.Normals.push_back(Normals[Indices[idx]].x);
		OutContext.Normals.push_back(Normals[Indices[idx]].y);
		OutContext.Normals.push_back(Normals[Indices[idx]].z);
	}
}

} // namespace RAPI