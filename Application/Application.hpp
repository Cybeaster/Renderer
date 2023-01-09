#include <cstdint>
#include <string>
#include <Path.hpp>
#include "SmartPtr.hpp"
#include <Types.hpp>

namespace RenderAPI
{
  class TRenderer;
}

class Application
{
public:
  static auto GetApplication()
  {
    if (!application)
    {
      application = RenderAPI::TTSharedPtr<Application>(new Application());
      return application;
    }
    else
    {
      return application;
    }
  }

  static auto GetShaderLocalPath()
  {
    return RootDirPath.concat(SimpleCubeShaderLocalPath.string());
  }
  static auto GetResourceDirectory()
  {
    return ResourceDirectory;
  }

  /**
   * @brief Programm start.
   * @details Initializes Renderer class.
   *
   */
  void Start(int argc, char **argv);

private:
  Application() = default;

  static inline RenderAPI::TTSharedPtr<Application> application = nullptr;

  static inline TPath DebugPath = current_path();
  static inline TPath RootDirPath = current_path().parent_path();

  static inline TPath SimpleCubeShaderLocalPath = "\\Externals\\Shaders\\SimpleCube.shader";
  static inline TPath ResourceDirectory = "\\Externals\\Resources\\";
};