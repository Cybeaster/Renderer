#include "Shader.hpp"
#include <Renderer.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <GL/glew.h>

namespace RenderAPI
{
    Shader::Shader(const String source)
    {
        Init(source);
    }

    Shader::~Shader()
    {
        GLCall(glDeleteProgram(rendererID));
    }

    void Shader::Init(const String source)
    {
        filePath = source;
        ShaderSource shaderSource = ParseShader(source);
        rendererID = CreateShader(shaderSource.vertexShader, shaderSource.fragmentShader);
    }

    void Shader::SetUniform1i(const String name, int32_t v0)
    {
        GLCall(glUniform1i(GetUnformLocation(name), v0));
    }

    void Shader::SetUniform1f(const String name, float v0)
    {
        GLCall(glUniform1f(GetUnformLocation(name), v0));
    }

    uint32 Shader::GetUnformLocation(const String &name)
    {
        if (locationCache.find(name) != locationCache.end())
            return locationCache[name];

        GLCall(int32_t location = glGetUniformLocation(rendererID, name.c_str()));
        if (location == -1)
            std::cout << "Warning unform " << name << "  doesnt exist." << std::endl;

        locationCache[name] = location;
        return location;
    }
    void Shader::SetUnformMat4f(const String name, Mat4 &matrix)
    {
        GLCall(glUniformMatrix4fv(GetUnformLocation(name), 1, GL_FALSE, &matrix[0][0]));
    }

    void Shader::Bind() const
    {
        GLCall(glUseProgram(rendererID));
    }

    void Shader::Unbind() const
    {
        GLCall(glUseProgram(0));
    }

    void Shader::SetUniform4f(const String name, float v0, float v1, float v2, float v3)
    {
        GLCall(glUniform4f(GetUnformLocation(name), v0, v1, v2, v3));
    }

    uint32 Shader::CompileShader(uint32 type, const String &source)
    {
        uint32 id = glCreateShader(type);
        const char *src = source.c_str();
        glShaderSource(id, 1, &src, nullptr);
        glCompileShader(id);

        int32_t result;
        glGetShaderiv(id, GL_COMPILE_STATUS, &result);
        if (result == GL_FALSE)
        {
            int32_t lenght;
            glGetShaderiv(id, GL_INFO_LOG_LENGTH, &lenght);
            char *message = (char *)alloca(lenght * sizeof(char));
            glGetShaderInfoLog(id, lenght, &lenght, message);
            std::cout << "Shaders isnt compiled" << std::endl;
            std::cout << message << std::endl;
            glDeleteShader(id);
            return GL_FALSE;
        }
        return id;
    }

    ShaderSource Shader::ParseShader(const String &filePath)
    {
        std::fstream stream(filePath);
        if (stream.is_open())
        {
            ShaderType currentType = ShaderType::NONE;
            String line;
            std::stringstream stringStream[2];

            while (getline(stream, line))
            {
                if (line.find("#shader") != String::npos)
                {
                    if (line.find("vertex") != String::npos)
                    {
                        currentType = ShaderType::VERTEX;
                    }
                    else if (line.find("fragment") != String::npos)
                    {
                        currentType = ShaderType::FRAGMENT;
                    }
                }
                else
                {
                    stringStream[int(currentType)] << line << '\n';
                }
            }
            stream.close();
            return {stringStream[0].str(), stringStream[1].str()};
        }

        return {};
    }

    int Shader::CreateShader(const String &vertexShader, const String &fragmentShader)
    {
        uint32 program = glCreateProgram();
        uint32 vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
        uint32 fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);
        glAttachShader(program, vs);
        glAttachShader(program, fs);
        glLinkProgram(program);
        glValidateProgram(program);

        glDeleteShader(vs);
        glDeleteShader(fs);

        return program;
    }

} // RenderAPI
