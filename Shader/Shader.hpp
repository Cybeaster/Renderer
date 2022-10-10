#pragma once
#include <RenderAPI.hpp>
#include <unordered_map>

struct ShaderSource
{
    String vertexShader;
    String fragmentShader;
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
        void Init(const String source);

        void Bind() const;
        void Unbind() const;
        void SetUniform4f(const String name, float v0, float v1, float v2, float v3);

        void SetUniform1f(const String name, float v0);
        void SetUniform1i(const String name, int32_t v0);
        void SetUnformMat4f(const String name, Mat4 &matrix);

        Shader() = default;
        Shader(const String source);
        ~Shader();

    private:
        uint32 CompileShader(uint32 type, const String &source);
        uint32 GetUnformLocation(const String &name);
        int CreateShader(const String &vertexShader, const String &fragmentShader);
        ShaderSource ParseShader(const String &filePath);
        bool CompileShader();

        String filePath;
        uint32 rendererID = 277;
        std::unordered_map<String, int32_t> locationCache;
    };
} // RenderAPI
