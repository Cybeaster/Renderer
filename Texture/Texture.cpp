#include "Texture.hpp"

#include "SOIL2.h"

#include <iostream>

OTexture::OTexture(const OPath& path, bool IsAF_Enabled)
{
	EnableAF = IsAF_Enabled && glewIsSupported("GL_EXT_texture_filter_anisotropic");
	textureID = SOIL_load_OGL_texture(
	    path.string().c_str(),
	    SOIL_LOAD_AUTO,
	    SOIL_CREATE_NEW_ID,
	    SOIL_FLAG_INVERT_Y);
	if (textureID == 0)
		std::cerr << "Couldn't find texture file" << path << std::endl;
}

OTexture::~OTexture()
{
	GLCall(glDeleteTextures(1, &textureID));
}

void OTexture::Bind(uint32 slot) const
{
	GLCall(glActiveTexture(GL_TEXTURE0 + slot));
	GLCall(glBindTexture(GL_TEXTURE_2D, textureID));
	// GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
	//  GLCall(glGenerateMipmap(GL_TEXTURE_2D)); // generate mipmap

	if (EnableAF)
	{
		float anisoSetting = 0.0f;
		glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &anisoSetting);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisoSetting);
	}
}

void OTexture::Unbind()
{
	GLCall(glBindTexture(GL_TEXTURE_2D, 0));
}