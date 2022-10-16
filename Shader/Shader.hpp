#pragma once
#include <unordered_map>
#include "Math.hpp"
#include "Vector.hpp"
#include "Types.hpp"
struct ShaderSource
{
    TString vertexShader;
    TString fragmentShader;
};

enum class ShaderType
{
    NONE = -1,
    VERTEX = 0,
    FRAGMENT = 1
};

namespace RenderAPI
{
    class Shader
    {
    public:
        void Init(const TString source);

        void Bind() const;
        void Unbind() const;
        void SetUniform4f(const TString name, float v0, float v1, float v2, float v3);

        void SetUniform1f(const TString name, float v0);
        void SetUniform1i(const TString name, int32_t v0);
        void SetUnformMat4f(const TString name, TMat4 &matrix);

        Shader() = default;
        Shader(const TString source);
        ~Shader();

    private:
        uint32 CompileShader(uint32 type, const TString &source);
        uint32 GetUnformLocation(const TString &name);
        int CreateShader(const TString &vertexShader, const TString &fragmentShader);
        ShaderSource ParseShader(const TString &filePath);
        bool CompileShader();

        TString filePath;
        uint32 rendererID = 277;
        std::unordered_map<TString, int32_t> locationCache;
    };
} // RenderAPI
