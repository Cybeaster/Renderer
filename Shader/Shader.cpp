#include "Shader.hpp"
#include <Renderer.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <GL/glew.h>



Shader::Shader(const std::string source)
    :m_FilePath(source), m_RendererID(0)
{
    ShaderSource shaderSource = ParseShader(source);
    m_RendererID = CreateShader(shaderSource.vertexShader,shaderSource.fragmentShader);
}



Shader::~Shader()
{
    GLCall(glDeleteProgram(m_RendererID));
}


void Shader::SetUniform1i(const std::string name, int32_t v0)
{
    GLCall(glUniform1i(GetUnformLocation(name),v0));
}

void Shader::SetUniform1f(const std::string name, float v0)
{
    GLCall(glUniform1f(GetUnformLocation(name),v0));
}

uint32_t Shader::GetUnformLocation(const std::string& name)
{
    if(m_LocationCache.find(name) != m_LocationCache.end())
        return m_LocationCache[name];

    GLCall(int32_t location = glGetUniformLocation(m_RendererID,name.c_str()));
    if(location == -1)
        std::cout<<"Warning unform "<<name<<"  doesnt exist."<<std::endl;
        
    m_LocationCache[name] = location;
    return location;
}
void Shader::SetUnformMat4f(const std::string name, glm::mat4& matrix)
{
    GLCall(glUniformMatrix4fv(GetUnformLocation(name),1,GL_FALSE,&matrix[0][0]));
}


void Shader::Bind()const
{
    GLCall(glUseProgram(m_RendererID));
}

void Shader::Unbind() const
{
    GLCall(glUseProgram(0));
}

void Shader::SetUniform4f(const std::string name, float v0, float v1, float v2, float v3)
{
    GLCall(glUniform4f(GetUnformLocation(name),v0,v1,v2,v3));
}



uint32_t Shader::CompileShader( uint32_t type,const std::string& source)
{
    uint32_t id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id,1,&src,nullptr);
    glCompileShader(id);


    int32_t result;
    glGetShaderiv(id,GL_COMPILE_STATUS,&result);
    if(result == GL_FALSE)
    {
        int32_t lenght;
        glGetShaderiv(id,GL_INFO_LOG_LENGTH,&lenght);
        char* message = (char*)alloca(lenght * sizeof(char));
        glGetShaderInfoLog(id,lenght,&lenght,message);
        std::cout<<"Shaders isnt compiled"<<std::endl;
        std::cout<<message<<std::endl;
        glDeleteShader(id);
        return GL_FALSE;
    }
    return id;
}


ShaderSource Shader::ParseShader(const std::string& filePath)
{
    std::fstream stream(m_FilePath);
    if(stream.is_open())
    {
        ShaderType currentType = ShaderType::NONE;
        std::string line;
        std::stringstream ss[2];

        while(getline(stream,line))
        {
            if(line.find("#shader") != std::string::npos)
            {
                if(line.find("vertex") != std::string::npos)
                {
                    currentType = ShaderType::VERTEX;
                }
                else if(line.find("fragment") != std::string::npos)
                {
                    currentType = ShaderType::FRAGMENT;
                }
            }
            else
            {
                ss[int(currentType)] << line << '\n';
            }
        }
        return {ss[0].str() , ss[1].str()};
    }

    return {};
}



int Shader::CreateShader(const std::string& vertexShader, const std::string& fragmentShader)
{
    uint32_t program = glCreateProgram();
    uint32_t vs = CompileShader(GL_VERTEX_SHADER,vertexShader);
    uint32_t fs = CompileShader(GL_FRAGMENT_SHADER,fragmentShader);
    glAttachShader(program,vs);
    glAttachShader(program,fs);
    glLinkProgram(program);
    glValidateProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);
    
    return program;
}
