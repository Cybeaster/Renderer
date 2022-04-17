#pragma once
#include <Renderer.hpp>

 class Texture
 {

public:

    Texture(const std::string path);
    ~Texture();

    void Bind(uint32_t slot = 0)const;
    void Unbind();

    
private:

    uint32_t textureID;
};
