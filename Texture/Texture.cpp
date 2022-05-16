#include "Texture.hpp"
#include <iostream>
#include "SOIL2.h"

 Texture::Texture(const std::string path)
 {
   textureID = SOIL_load_OGL_texture(
          path.c_str(),
          SOIL_LOAD_AUTO,
          SOIL_CREATE_NEW_ID,
          SOIL_FLAG_INVERT_Y);
        if(textureID == 0)
            std::cerr<<"Couldn't find texture file" << path <<std::endl;
 }
 
 Texture::~Texture()
 {
     GLCall(glDeleteTextures(1,&textureID));
 }
 

 
void Texture::Bind(uint32_t slot)const
{
    GLCall(glActiveTexture(GL_TEXTURE0 + slot));
    GLCall(glBindTexture(GL_TEXTURE_2D,textureID));
}

void Texture::Unbind()
{
    GLCall(glBindTexture(GL_TEXTURE_2D,0));
}