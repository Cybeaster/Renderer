#include "InputHandlers/InputHandler.hpp"
#include "Profiler/Profiler.hpp"
#include "ShaderName.hpp"
#include "SmartPtr.hpp"

#include <Path.hpp>
#include <Types.hpp>
#include <cstdint>
#include <string>

namespace RAPI
{

class OApplication
{
public:
	static auto GetApplication()
	{
		if (!Application)
		{
			Application = RAPI::OSharedPtr<OApplication>(new OApplication());
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

	void ParseInput(int argc, char** argv);
	void SetupInput();
	void StartProgram();

	OInputHandler InputHandler;

	static inline RAPI::OSharedPtr<OApplication> Application = nullptr;

	static inline OPath DebugPath = current_path();
	static inline OPath RootDirPath = current_path();

	static inline OPath ShadersDir = R"(\Externals\Shaders\)";
	static inline OPath ResourceDirectory = R"(\Externals\Resources\)";

	static inline SShaderName SimpleCubeShader = "\\SimpleCube.shader";
	static inline SShaderName SimpleTextureShader = "\\SimpleTexture.shader";
};
} // namespace RAPI