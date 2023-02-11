#include "Shader.hpp"

#include <GL/glew.h>
#include <Renderer.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

namespace RenderAPI
{
TShader::TShader(const OPath Source)
{
	Init(Source);
}

TShader::~TShader()
{
	GLCall(glDeleteProgram(RendererID));
}

void TShader::Init(const OPath Source)
{
	FilePath = Source;
	ShaderSource shaderSource = ParseShader(Source);
	RendererID = CreateShader(shaderSource.vertexShader, shaderSource.fragmentShader);
}

void TShader::SetUniform1i(const OString name, int32_t v0)
{
	GLCall(glUniform1i(GetUnformLocation(name), v0));
}

void TShader::SetUniform1f(const OString name, float v0)
{
	GLCall(glUniform1f(GetUnformLocation(name), v0));
}

uint32 TShader::GetUnformLocation(const OString& name)
{
	if (LocationCache.find(name) != LocationCache.end())
		return LocationCache[name];

	GLCall(int32_t location = glGetUniformLocation(RendererID, name.c_str()));
	if (location == -1)
		std::cout << "Warning unform " << name << "  doesnt exist." << std::endl;

	LocationCache[name] = location;
	return location;
}
void TShader::SetUnformMat4f(const OString name, OMat4&& matrix)
{
	GLCall(glUniformMatrix4fv(GetUnformLocation(name), 1, GL_FALSE, &matrix[0][0]));
}

void TShader::SetUnformMat4f(const OString name, const OMat4& matrix)
{
	GLCall(glUniformMatrix4fv(GetUnformLocation(name), 1, GL_FALSE, &matrix[0][0]));
}

void TShader::Bind() const
{
	GLCall(glUseProgram(RendererID));
}

void TShader::Unbind() const
{
	GLCall(glUseProgram(0));
}

void TShader::SetUniform4f(const OString name, float v0, float v1, float v2, float v3)
{
	GLCall(glUniform4f(GetUnformLocation(name), v0, v1, v2, v3));
}

uint32 TShader::CompileShader(uint32 type, const OString& Source)
{
	uint32 id = glCreateShader(type);
	const char* src = Source.c_str();
	glShaderSource(id, 1, &src, nullptr);
	glCompileShader(id);

	int32_t result;
	glGetShaderiv(id, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE)
	{
		int32_t lenght;
		glGetShaderiv(id, GL_INFO_LOG_LENGTH, &lenght);
		char* message = (char*)alloca(lenght * sizeof(char));
		glGetShaderInfoLog(id, lenght, &lenght, message);
		std::cout << "Shaders isnt compiled" << std::endl;
		std::cout << message << std::endl;
		glDeleteShader(id);
		return GL_FALSE;
	}
	return id;
}

ShaderSource TShader::ParseShader(const OPath& filePath)
{
	std::fstream stream(filePath);
	if (stream.is_open())
	{
		ShaderType currentType = ShaderType::NONE;
		OString line;
		std::stringstream stringStream[2];

		while (getline(stream, line))
		{
			if (line.find("#shader") != OString::npos)
			{
				if (line.find("vertex") != OString::npos)
				{
					currentType = ShaderType::VERTEX;
				}
				else if (line.find("fragment") != OString::npos)
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
		return { stringStream[0].str(), stringStream[1].str() };
	}

	return {};
}

int TShader::CreateShader(const OString& vertexShader, const OString& fragmentShader)
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

} // namespace RenderAPI
