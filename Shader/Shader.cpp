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
	GLCall(glProgramUniform1i(RendererID, GetUniformLocation(name), v0));
}

void OShader::SetUniform1ui(const OString& name, uint32 v0)
{
	GLCall(glProgramUniform1ui(RendererID, GetUniformLocation(name), v0));
}

void OShader::SetUniform1f(const OString& name, float v0)
{
	GLCall(glProgramUniform1f(RendererID, GetUniformLocation(name), v0));
}

uint32 OShader::GetUniformLocation(const OString& name)
{
	if (LocationCache.find(name) != LocationCache.end())
		return LocationCache[name];

	GLCall(int32 location = glGetUniformLocation(RendererID, name.c_str()));
	if (location == -1)
		std::cout << "Warning uniform " << name << "  doesn't exist." << std::endl;

	LocationCache[name] = location;
	return location;
}

void OShader::SetUniformVec3f(const OString& name, const OVec3& Vector)
{
	GLCall(glProgramUniform3fv(RendererID, GetUniformLocation(name), 1, glm::value_ptr(Vector)));
}

void OShader::SetUniformVec4f(const OString& name, const OVec4& Vector)
{
	GLCall(glProgramUniform4fv(RendererID, GetUniformLocation(name), 1, glm::value_ptr(Vector)));
}

void OShader::SetUniform4f(const OString& name, float v0, float v1, float v2, float v3)
{
	GLCall(glProgramUniform4f(RendererID, GetUniformLocation(name), v0, v1, v2, v3));
}

void OShader::SetUniformMat4f(const OString& name, const OMat4& matrix)
{
	GLCall(glProgramUniformMatrix4fv(RendererID, GetUniformLocation(name), 1, GL_FALSE, glm::value_ptr(matrix)));
}

void OShader::Bind() const
{
	GLCall(glUseProgram(RendererID));
}

void OShader::Unbind() const
{
	GLCall(glUseProgram(0));
}

uint32 OShader::CompileShader(uint32 type, const OString& Source)
{
	uint32 id = glCreateShader(type);
	const char* src = Source.c_str();
	GLCall(glShaderSource(id, 1, &src, nullptr));
	GLCall(glCompileShader(id));

	if (!CatchShaderError(GL_COMPILE_STATUS, id))
	{
		glDeleteShader(id);
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
	GLCall(uint32 program = glCreateProgram());
	uint32 vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
	uint32 fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);
	GLCall(glAttachShader(program, vs));
	GLCall(glAttachShader(program, fs));
	glLinkProgram(program);

	if (!CatchShaderError(GL_LINK_STATUS, program))
	{
		glDeleteProgram(program);
	}

	glValidateProgram(program);
	if (!CatchShaderError(GL_VALIDATE_STATUS, program))
	{
		glDeleteProgram(program);
	}

	GLCall(glDeleteShader(vs));
	GLCall(glDeleteShader((fs)));

	return program;
}

bool OShader::CatchShaderError(int32 ErrorType, uint32 RenderID)
{
	int32 isOk;
	glGetProgramiv(RenderID, ErrorType, &isOk);

	if (isOk == GL_FALSE)
	{
		int32_t length;
		glGetProgramiv(RenderID, GL_INFO_LOG_LENGTH, &length);

		OString msg;
		msg.resize(length);

		glGetProgramInfoLog(RenderID, length, &length, msg.data());

		std::cout << "Shader Error!" << std::endl;
		std::cout << msg << std::endl;
	}
	return isOk;
}

} // namespace RAPI
