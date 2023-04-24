#include "InputHandlers/InputHandler.hpp"
#include "ModelsImporter/ImportManager.hpp"
#include "NamedThreadPool/NamedThreadPool.hpp"
#include "Profiler/Profiler.hpp"
#include "Renderer.hpp"
#include "ShaderName.hpp"
#include "SmartPtr.hpp"
#include "Window/Window.hpp"

#include <Path.hpp>
#include <Types.hpp>
#include <cstdint>
#include <string>

namespace RAPI
{

class OApplication
{
public:
	static OApplication* GetApplication();
	static OString GetShaderLocalPathWith(const SShaderName& Name);
	static OString GetResourceDirectoryWith(const OPath& Path);

	/**
	 * @brief Programm start.
	 *
	 */
	void Start(int argc, char** argv);

private:
	OApplication() = default;

	void CreateWindow();
	void ParseInput(int argc, char** argv);
	void SetupInput();
	void InitRenderer();
	void InitConsoleInput();

	NODISCARD SRenderContext MakeRendererContext() const;

	OInputHandler InputHandler;
	OUniquePtr<OWindow> Window;
	ONamedThreadPool NamedThreadPool;
	OImportManager Importer;
	static inline OPath DebugPath = current_path();
	static inline OPath RootDirPath = current_path();

	static inline OPath ShadersDir = R"(\Externals\Shaders\)";
	static inline OPath ResourceDirectory = R"(\Externals\Resources\)";

	static inline SShaderName SimpleCubeShader = "\\SimpleCube.shader";
	static inline SShaderName SimpleTextureShader = "\\SimpleTexture.shader";
	static inline SShaderName BasicShader = "\\Basic.shader";
	static inline OApplication* Application = nullptr;
};
} // namespace RAPI