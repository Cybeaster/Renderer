#pragma once
#include "Math.hpp"
#include "Types.hpp"
#include "Vector.hpp"

#include <Path.hpp>
#include <unordered_map>

struct SHaderSource
{
	OString vertexShader;
	OString fragmentShader;
};

enum class ShaderType
{
	NONE = -1,
	VERTEX = 0,
	FRAGMENT = 1
};

namespace RAPI
{
class OShader
{
public:
	void Init(const OPath& source);

	void Bind() const;
	void Unbind() const;
	void SetUniform4f(const OString& name, float v0, float v1, float v2, float v3);

	void SetUniform1f(const OString& name, float v0);
	void SetUniform1i(const OString& name, int32 v0);
	void SetUniform1ui(const OString& name, uint32 v0);


	void SetUniformMat4f(const OString& name, const OMat4& matrix);
	void SetUniformVec3f(const OString& name, const OVec3& Vector);
	void SetUniformVec4f(const OString& name, const OVec4& Vector);
	OShader() = default;
	explicit OShader(const OPath& source);
	~OShader();

private:

	bool CatchShaderError(int32 ErrorType, uint32 RenderID);

	uint32 CompileShader(uint32 type, const OString& source);
	uint32 GetUniformLocation(const OString& name);
	int CreateShader(const OString& vertexShader, const OString& fragmentShader);
	SHaderSource ParseShader(const OPath& filePath);
	bool CompileShader();

	OPath FilePath;
	uint32 RendererID = 277;
	std::unordered_map<OString, int32_t> LocationCache;
};
} // namespace RAPI
