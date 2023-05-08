#include "Shader.hpp"

#include <GL/glew.h>
#include <Renderer.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

namespace RAPI
{
OShader::OShader(const OPath& Source)
{
	Init(Source);
}

OShader::~OShader()
{
	GLCall(glDeleteProgram(RendererID));
}

void OShader::Init(const OPath& Source)
{
	FilePath = Source;
	SHaderSource shaderSource = ParseShader(Source);
	RendererID = CreateShader(shaderSource.vertexShader, shaderSource.fragmentShader);
}

void OShader::SetUniform1i(const OString& name, int32 v0)
{
	GLCall(glUniform1i(GetUniformLocation(name), v0));
}

void OShader::SetUniform1f(const OString& name, float v0)
{
	GLCall(glUniform1f(GetUniformLocation(name), v0));
}

uint32 OShader::GetUniformLocation(const OString& name)
{
	if (LocationCache.find(name) != LocationCache.end())
		return LocationCache[name];

	GLCall(int32 location = glGetUniformLocation(RendererID, name.c_str()));
	if (location == -1)
		std::cout << "Warning unform " << name << "  doesn't exist." << std::endl;

	LocationCache[name] = location;
	return location;
}

void OShader::SetUniformVec3f(const OString& name, const OVec3& Vector)
{
	GLCall(glUniformMatrix3fv(GetUniformLocation(name), 1, GL_FALSE, glm::value_ptr(Vector)));
}

void OShader::SetUniformVec4f(const OString& name, const OVec4& Vector)
{
	GLCall(glUniformMatrix4fv(GetUniformLocation(name), 1, GL_FALSE, glm::value_ptr(Vector)));
}

void OShader::SetUniformMat4f(const OString& name, const OMat4& matrix)
{
	GLCall(glUniformMatrix4fv(GetUniformLocation(name), 1, GL_FALSE, glm::value_ptr(matrix)));
}

void OShader::Bind() const
{
	GLCall(glUseProgram(RendererID));
}

void OShader::Unbind() const
{
	GLCall(glUseProgram(0));
}

void OShader::SetUniform4f(const OString& name, float v0, float v1, float v2, float v3)
{
	GLCall(glUniform4f(GetUniformLocation(name), v0, v1, v2, v3));
}

uint32 OShader::CompileShader(uint32 type, const OString& Source)
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

SHaderSource OShader::ParseShader(const OPath& filePath)
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

int OShader::CreateShader(const OString& vertexShader, const OString& fragmentShader)
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

} // namespace RAPI
