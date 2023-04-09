#pragma once

#include "StatObject.hpp"
#include "TypeTraits.hpp"
#include "Vector.hpp"

namespace RAPI
{

template<typename TID = uint32>
struct SStatGroupID
{
	SStatGroupID() = default;
	~SStatGroupID() = default;

	explicit SStatGroupID(TID Other)
	    : ID(Other) {}

	TID ID = SInvalidValue<TID>::Get();

	bool operator==(const SStatGroupID& Other)
	{
		return ID == Other.ID;
	}
};

template<typename TID>
struct STStatGroupHash
{
	auto operator()(const SStatGroupID<TID>& Other) const
	{
		return GetHash(Other.ID);
	}
};

class OStatGroup
{
	explicit OStatGroup(OString&& GroupName);

public:
	OVector<OStatObject> StatObjects;
};

} // namespace RAPI
