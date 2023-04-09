#include "Torus.hpp"

void RAPI::OTorus::Init(int32 Precision) noexcept
{
	const auto sqrtPrecision = Precision * Precision;
	NumVertices = sqrtPrecision + 2 * Precision + 1;
}
void RAPI::OTorus::PreInit(int32 Precision) noexcept
{
	OModel::PreInit(Precision);
	STangents.resize(NumVertices);
	TTangents.resize(NumVertices);

	CalcFirstRing(Precision);
	MultiplyRings(Precision);
	CalcIndices(Precision);
}
void RAPI::OTorus::CalcFirstRing(uint32 Precision)
{
	for (size_t it = 0; it < Precision + 1; ++it)
	{
		// build the ring by rotating points around the origin, then moving outward
		const float amt = SMath::ToRadians(it * 360.F / Precision);
		OMat4 rMat = glm::rotate(OMat4(1.F), amt, OVec3(0, 0, 1.F));
		OVec3 startPos(rMat * OVec4(0.F, OuterRadius, 0.F, 1.F));
		Vertices[it] = OVec3(startPos + OVec3(InnerRadius, 0.F, 0.F));

		// Compute texture coords for each vertex on the ring
		TexCoords[it] = OVec2(0, static_cast<float>(it) / static_cast<float>(Precision));

		// compute tangents and normals - first tangent is Y-axis rotated around Z.
		rMat = glm::rotate(OMat4(1.F), amt + (SMath::Pi / 2.0F), OVec3(0, 0, 1.F));
		TTangents[it] = OVec3(rMat * OVec4(0, -1.F, 0, 0));
		STangents[it] = OVec3(OVec3(0, 0, -1.F));

		Normals[it] = glm::cross(TTangents[it], STangents[it]);
	}
}

void RAPI::OTorus::MultiplyRings(uint32 Precision)
{
	auto incPrecision = Precision + 1;
	for (size_t ringIdx = 1; ringIdx < Precision; ++ringIdx)
	{
		for (size_t vertIdx = 0; vertIdx < Precision; ++vertIdx)
		{
			// rotate the vertex positions of the original ring around the Y axis
			auto amt = static_cast<float>(SMath::ToRadians(ringIdx * 360.F / Precision));
			OMat4 rMat = glm::rotate(OMat4(1.F), amt, OVec3(0, 1.F, 0));

			Vertices[ringIdx * incPrecision + vertIdx] = OVec3(rMat * OVec4(Vertices[vertIdx], 1.F));

			// compute the texture coord for the vert on the new rings
			TexCoords[ringIdx * incPrecision + vertIdx] = OVec2((float)ringIdx * 2.F / (float)Precision, TexCoords[vertIdx].t); // NOLINT

			rMat = glm::rotate(OMat4(1.F), amt, OVec3(0, 1.F, 0));
			STangents[ringIdx * incPrecision + vertIdx] = OVec3(rMat * OVec4(STangents[vertIdx], 1.F));

			rMat = glm::rotate(OMat4(1.F), amt, OVec3(0.0, 1.F, 0.F));

			TTangents[ringIdx * incPrecision + vertIdx] = OVec3(rMat * OVec4(TTangents[vertIdx], 1.F));

			// rotate the normal vector around the Y axis
			rMat = glm::rotate(OMat4(1.F), amt, OVec3(0.0, 1.F, 0.F));
			Normals[ringIdx * incPrecision + vertIdx] = OVec3(rMat * OVec4(Normals[vertIdx], 1.F));
		}
	}
}
void RAPI::OTorus::CalcIndices(uint32 Precision)
{
	const auto incPrecision = Precision + 1;
	for (uint32 ringIdx = 0; ringIdx < Precision; ++ringIdx)
	{
		for (uint32 vertIdx = 0; vertIdx < Precision; ++vertIdx)
		{
			const auto idx = (ringIdx * Precision) + vertIdx;

			Indices[(idx * 2) * 3 + 0] = ringIdx * incPrecision + vertIdx;
			Indices[(idx * 2) * 3 + 1] = (ringIdx + 1) * incPrecision + vertIdx;
			Indices[(idx * 2) * 3 + 2] = ringIdx * incPrecision + vertIdx + 1;

			Indices[((idx * 2) + 1) * 3 + 0] = ringIdx * incPrecision + vertIdx + 1;
			Indices[((idx * 2) + 1) * 3 + 1] = (ringIdx + 1) * incPrecision + vertIdx;
			Indices[((idx * 2) + 1) * 3 + 2] = (ringIdx + 1) * incPrecision + vertIdx + 1;
		}
	}
}
void RAPI::OTorus::GetVertexTextureNormalPositions(SModelContext& OutContext)
{
	const auto numVertices = GetNumVertices();
	for (uint32 idx = 0; idx < numVertices; ++idx)
	{
		OutContext.VertexCoords.push_back(Vertices[idx].x);
		OutContext.VertexCoords.push_back(Vertices[idx].y);
		OutContext.VertexCoords.push_back(Vertices[idx].z);

		OutContext.TextureCoords.push_back(TexCoords[idx].s);
		OutContext.TextureCoords.push_back(TexCoords[idx].t);

		OutContext.NormalsCoords.push_back(Normals[idx].x);
		OutContext.NormalsCoords.push_back(Normals[idx].y);
		OutContext.NormalsCoords.push_back(Normals[idx].z);
	}
}
