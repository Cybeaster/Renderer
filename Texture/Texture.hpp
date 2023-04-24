#pragma once

#include "Path.hpp"
#include "Types.hpp"

#include <string>

namespace RAPI
{
class OTexture
{
public:
	OTexture(const OPath& path, bool IsAF_Enabled = false);
	~OTexture();

	void Bind(uint32 slot = 0) const;
	void Unbind();

private:
	bool EnableAF = false;
	uint32 textureID;
};

} // namespace RAPI
