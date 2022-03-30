#pragma once
#include <string>
#include <unordered_map>
struct ShaderSource
{
    std::string vertexShader;
    std::string fragmentShader;
};

enum class ShaderType
{
    NONE = -1,
    VERTEX = 0,
    FRAGMENT = 1
};

class Shader
{
public:
    void Bind()const;
    void Unbind() const;
    void SetUniform4f(const std::string name, float v0, float v1, float v2, float v3);
    
    void SetUniform1f(const std::string name, float v0);
    void SetUniform1i(const std::string name, int32_t v0);

    Shader(const std::string source);
    ~Shader();
private:
   uint32_t CompileShader( uint32_t type,const std::string& source);
   int CreateShader(const std::string& vertexShader, const std::string& fragmentShader);
   ShaderSource ParseShader(const std::string& filePath);
   bool CompileShader();


   uint32_t GetUnformLocation(const std::string& name);

   std::string m_FilePath;
   uint32_t m_RendererID;
   
   std::unordered_map<std::string,int32_t> m_LocationCache;

};

