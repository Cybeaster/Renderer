#include "Torus.hpp"

void RAPI::OTorus::Init(uint32 Precision) noexcept
{
}
void RAPI::OTorus::PreInit(uint32 Precision) noexcept
{
	Super::PreInit(Precision);
	for (int i = 0; i < NumVertices; i++)
	{
		STangents.emplace_back();
	}
	for (int i = 0; i < NumVertices; i++)
	{
		TTangents.emplace_back();
	}

	CalcFirstRing(Precision);
	MultiplyRings(Precision);
	CalcIndices(Precision);
}
void RAPI::OTorus::CalcFirstRing(uint32 Precision)
{
	// calculate first ring
	for (int i = 0; i < Precision + 1; i++)
	{
		float amt = SMath::ToRadians(i * 360.0f / Precision);
		// build the ring by rotating points around the origin, then moving them outward
		glm::mat4 rMat = glm::rotate(glm::mat4(1.0f), amt, glm::vec3(0.0f, 0.0f, 1.0f));
		glm::vec3 initPos(rMat * glm::vec4(0.0f, OuterRadius, 0.0f, 1.0f));
		Vertices[i] = glm::vec3(initPos + glm::vec3(InnerRadius, 0.0f, 0.0f));
		// compute texture coordinates for each vertex on the ring
		TexCoords[i] = glm::vec2(0.0f, ((float)i / (float)Precision));
		// compute tangents and normals -- first tangent is Y-axis rotated around Z
		rMat = glm::rotate(glm::mat4(1.0f), amt + (3.14159f / 2.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		TTangents[i] = glm::vec3(rMat * glm::vec4(0.0f, -1.0f, 0.0f, 1.0f));
		STangents[i] = glm::vec3(glm::vec3(0.0f, 0.0f, -1.0f)); // second tangent is -Z.
		Normals[i] = glm::cross(TTangents[i], STangents[i]); // their X-product is the normal.
	}
}

void RAPI::OTorus::MultiplyRings(uint32 Precision)
{
	// rotate the first ring about Y to get the other rings
	for (int ring = 1; ring < Precision + 1; ring++)
	{
		for (int vert = 0; vert < Precision + 1; vert++)
		{
			// rotate the vertex positions of the original ring around the Y axis
			float amt = (float)(SMath::ToRadians(ring * 360.0f / Precision));
			glm::mat4 rMat = glm::rotate(glm::mat4(1.0f), amt, glm::vec3(0.0f, 1.0f, 0.0f));
			Vertices[ring * (Precision + 1) + vert] = glm::vec3(rMat * glm::vec4(Vertices[vert], 1.0f));
			// compute the texture coordinates for the vertices in the new rings
			TexCoords[ring * (Precision + 1) + vert] = glm::vec2((float)ring * 2.0f / (float)Precision, TexCoords[vert].t);
			// rotate the tangent and bitangent vectors around the Y axis
			rMat = glm::rotate(glm::mat4(1.0f), amt, glm::vec3(0.0f, 1.0f, 0.0f));
			STangents[ring * (Precision + 1) + vert] = glm::vec3(rMat * glm::vec4(STangents[vert], 1.0f));
			rMat = glm::rotate(glm::mat4(1.0f), amt, glm::vec3(0.0f, 1.0f, 0.0f));
			TTangents[ring * (Precision + 1) + vert] = glm::vec3(rMat * glm::vec4(TTangents[vert], 1.0f));
			// rotate the normal vector around the Y axis
			rMat = glm::rotate(glm::mat4(1.0f), amt, glm::vec3(0.0f, 1.0f, 0.0f));
			Normals[ring * (Precision + 1) + vert] = glm::vec3(rMat * glm::vec4(Normals[vert], 1.0f));
		}
	}
}
void RAPI::OTorus::CalcIndices(uint32 Precision)
{
	const auto incPrecision = Precision + 1;
	for (int ring = 0; ring < Precision; ring++)
	{
		for (int vert = 0; vert < Precision; vert++)
		{
			Indices[((ring * Precision + vert) * 2) * 3 + 0] = ring * (Precision + 1) + vert;
			Indices[((ring * Precision + vert) * 2) * 3 + 1] = (ring + 1) * (Precision + 1) + vert;
			Indices[((ring * Precision + vert) * 2) * 3 + 2] = ring * (Precision + 1) + vert + 1;
			Indices[((ring * Precision + vert) * 2 + 1) * 3 + 0] = ring * (Precision + 1) + vert + 1;
			Indices[((ring * Precision + vert) * 2 + 1) * 3 + 1] = (ring + 1) * (Precision + 1) + vert;
			Indices[((ring * Precision + vert) * 2 + 1) * 3 + 2] = (ring + 1) * (Precision + 1) + vert + 1;
		}
	}
}
void RAPI::OTorus::GetVertexTextureNormalPositions(SModelContext& OutContext)
{
	const auto numVertices = GetNumVertices();
	for (uint32 idx = 0; idx < numVertices; ++idx)
	{
		OutContext.Vertices.push_back(Vertices[idx].x);
		OutContext.Vertices.push_back(Vertices[idx].y);
		OutContext.Vertices.push_back(Vertices[idx].z);

		OutContext.TexCoords.push_back(TexCoords[idx].s);
		OutContext.TexCoords.push_back(TexCoords[idx].t);

		OutContext.Normals.push_back(Normals[idx].x);
		OutContext.Normals.push_back(Normals[idx].y);
		OutContext.Normals.push_back(Normals[idx].z);
	}
}
