#include "Texture.hpp"
#include <iostream>
#include "SOIL2.h"

TTexture::TTexture(const TPath& path)
{
    textureID = SOIL_load_OGL_texture(
        path.string().c_str(),
        SOIL_LOAD_AUTO,
        SOIL_CREATE_NEW_ID,
        SOIL_FLAG_INVERT_Y);
    if (textureID == 0)
        std::cerr << "Couldn't find texture file" << path << std::endl;
}

TTexture::~TTexture()
{
    GLCall(glDeleteTextures(1, &textureID));
}

void TTexture::Bind(uint32 slot) const
{
    GLCall(glActiveTexture(GL_TEXTURE0 + slot));
    GLCall(glBindTexture(GL_TEXTURE_2D, textureID));
}

void TTexture::Unbind()
{
    GLCall(glBindTexture(GL_TEXTURE_2D, 0));
}