#pragma once
#include <Renderer.hpp>
#include <string>
class TTexture
{

public:
    TTexture(const TPath& path);
    ~TTexture();

    void Bind(uint32 slot = 0) const;
    void Unbind();

private:
    uint32 textureID;
};
