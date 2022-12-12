#include <cstdint>
#include <string>
#include <filesystem>
#include "SmartPtr.hpp"
#include <Types.hpp>

namespace RenderAPI
{
  class Renderer;
}
using namespace std::filesystem;

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
    return RootDirPath.string() + SimpleCubeShaderLocalPath;
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

  static inline path DebugPath = current_path();
  static inline path RootDirPath = current_path().parent_path().parent_path();

  static inline TString SimpleCubeShaderLocalPath = "/Externals/Shaders/SimpleCube.shader";
};