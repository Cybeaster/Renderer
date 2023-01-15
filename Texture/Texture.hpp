#pragma once
#include <Renderer.hpp>
#include <string>
class TTexture
{

public:
    TTexture(const TPath& path, bool IsAF_Enabled = false);
    ~TTexture();

    void Bind(uint32 slot = 0) const;
    void Unbind();

private:

    bool EnableAF = false;
    uint32 textureID;
};
