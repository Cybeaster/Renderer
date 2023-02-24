#include "SmartPtr.hpp"

#include <Path.hpp>
#include <Types.hpp>
#include <cstdint>
#include <string>

namespace RenderAPI
{
class ORenderer;
}

struct SShaderName
{
	SShaderName() = default;
	SShaderName(const SShaderName& Str) = default;

	explicit SShaderName(const OString& Str) // NOLINT
	    : Name(Str)
	{
	}

	SShaderName(OString&& Str) noexcept // NOLINT
	    : Name(Move(Str))
	{
	}

	SShaderName(const char* Str) // NOLINT
	    : Name(Str)
	{
	}

	SShaderName(SShaderName&& Str) noexcept
	    : Name(Move(Str)) {}

	SShaderName& operator=(const OString& Str)
	{
		Name = Str;
		return *this;
	}

	SShaderName& operator=(const char* const Str)
	{
		Name = Str;
		return *this;
	}

	explicit operator OString() const
	{
		return Name;
	}

	OString Name;
};

class OApplication
{
public:
	static auto GetApplication()
	{
		if (!Application)
		{
			Application = RenderAPI::OTSharedPtr<OApplication>(new OApplication());
			return Application;
		}

		return Application;
	}

	static auto GetShaderLocalPathWith(const SShaderName& Name)
	{
		return RootDirPath.string() + ShadersDir.string() + Name.Name;
	}

	static auto GetResourceDirectoryWith(const OPath& Path)
	{
		return RootDirPath.string() + ResourceDirectory.string() + Path.string();
	}

	/**
	 * @brief Programm start.
	 * @details Initializes Renderer class.
	 *
	 */
	void Start(int argc, char** argv);

private:
	OApplication() = default;

	static inline RenderAPI::OTSharedPtr<OApplication> Application = nullptr;

	static inline OPath DebugPath = current_path();
	static inline OPath RootDirPath = current_path();

	static inline OPath ShadersDir = R"(\Externals\Shaders\)";
	static inline OPath ResourceDirectory = R"(\Externals\Resources\)";

	static inline SShaderName SimpleCubeShader = "\\SimpleCube.shader";
	static inline SShaderName SimpleTextureShader = "\\SimpeTexture.shader";
};