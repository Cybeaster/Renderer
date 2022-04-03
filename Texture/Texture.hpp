#pragma once
#include <Renderer.hpp>

 class Texture
 {

public:

    Texture(const std::string path);
    ~Texture();

    void Bind(uint32_t slot = 0)const;
    void Unbind();

    inline int GetWidth()
    {return m_Width;}
private:

    uint32_t m_RendererID;
    std::string m_FilePath;
    uint8_t* m_LocalBuffer;
    int32_t m_Width, m_Height, m_BPP;


};
