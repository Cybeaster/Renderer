
#include "Sphere.hpp"

namespace RenderAPI
{

OSphere::OSphere(uint32 Precision)
{
	Init(Precision);
}

void OSphere::Init(int32 Precision) noexcept
{
	// Because the whole sphere has to be "closed";
	const auto incPrecision = Precision + 1;

	auto squaredPrecision = Precision * Precision;
	NumVertices = squaredPrecision + 2 * incPrecision;
	NumIndices = squaredPrecision * 6;

	Vertices.reserve(NumVertices);
	TexCoords.reserve(NumVertices);
	Normals.reserve(NumVertices);
	Indices.reserve(NumIndices);

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

OSphere::OSphere()
{
	Init(DefaultPrecision);
}

void OSphere::GetVertexTextureNormalPositions(OTVector<float>& OutVertex, OTVector<float>& OutTexture, OTVector<float>& OutNormals)
{
	const auto indices = GetNumIndices();

	for (int idx = 0; idx < indices; ++idx)
	{
		OutVertex.push_back(Vertices[Indices[idx]].x);
		OutVertex.push_back(Vertices[Indices[idx]].y);
		OutVertex.push_back(Vertices[Indices[idx]].z);

		OutTexture.push_back(TexCoords[Indices[idx]].s);
		OutTexture.push_back(TexCoords[Indices[idx]].t);

		OutNormals.push_back(Normals[Indices[idx]].x);
		OutNormals.push_back(Normals[Indices[idx]].y);
		OutNormals.push_back(Normals[Indices[idx]].z);
	}
}

} // namespace RenderAPI