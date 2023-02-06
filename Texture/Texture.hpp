#pragma once
#include <Renderer.hpp>
#include <string>
class OTexture
{
public:
	OTexture(const TPath& path, bool IsAF_Enabled = false);
	~OTexture();

	void Bind(uint32 slot = 0) const;
	void Unbind();

private:
	bool EnableAF = false;
	uint32 textureID;
};
