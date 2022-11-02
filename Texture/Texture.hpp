#pragma once
#include <Renderer.hpp>
#include <string>
class Texture
{

public:
    Texture(const TString path);
    ~Texture();

    void Bind(uint32 slot = 0) const;
    void Unbind();

private:
    uint32 textureID;
};
