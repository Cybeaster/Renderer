#pragma once

#ifndef RENDERAPI_MODEL_HPP
#define RENDERAPI_MODEL_HPP

#include "Math.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Vector.hpp"

struct SModelContext
{
	OTVector<float> VertexCoords;
	OTVector<float> TextureCoords;
	OTVector<float> NormalsCoords;
};

namespace RenderAPI
{

class OModel
{
public:
	OModel()
	{
		Init(DefaultPrecision);
	}

	explicit OModel(const uint32 Precision)
	{
		Init(Precision);
	}

	NODISCARD FORCEINLINE uint32 GetNumVertices() const
	{
		return NumVertices;
	}

	NODISCARD FORCEINLINE uint32 GetNumIndices() const
	{
		return NumIndices;
	}

	NODISCARD FORCEINLINE const OTVector<int32>& GetIndices() const
	{
		return Indices;
	}

	void GetVertexTextureNormalPositions(SModelContext& OutContext);

protected:
	virtual void Init(int32) noexcept = 0;

	const uint32 DefaultPrecision = 48;

	uint32 NumVertices;
	uint32 NumIndices;

	OTVector<int32> Indices;
	OTVector<OVec3> Vertices;
	OTVector<OVec2> TexCoords;
	OTVector<OVec3> Normals;
};
} // namespace RenderAPI

#endif // RENDERAPI_MODEL_HPP
