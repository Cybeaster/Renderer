#pragma once
#include <unordered_map>
#include "Math.hpp"
#include "Vector.hpp"
#include "Types.hpp"
#include <Path.hpp>
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
    class TShader
    {
    public:
        void Init(const TPath source);

        void Bind() const;
        void Unbind() const;
        void SetUniform4f(const TString name, float v0, float v1, float v2, float v3);

        void SetUniform1f(const TString name, float v0);
        void SetUniform1i(const TString name, int32_t v0);
        void SetUnformMat4f(const TString name, TMat4 &&matrix);
        void SetUnformMat4f(const TString name, const TMat4 &matrix);

        TShader() = default;
        TShader(const TPath source);
        ~TShader();

    private:
        uint32 CompileShader(uint32 type, const TString &source);
        uint32 GetUnformLocation(const TString &name);
        int CreateShader(const TString &vertexShader, const TString &fragmentShader);
        ShaderSource ParseShader(const TPath &filePath);
        bool CompileShader();

        TPath FilePath;
        uint32 RendererID = 277;
        std::unordered_map<TString, int32_t> LocationCache;
    };
} // RenderAPI
