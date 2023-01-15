#include <cstdint>
#include <string>
#include <Path.hpp>
#include "SmartPtr.hpp"
#include <Types.hpp>

namespace RenderAPI
{
  class TRenderer;
}

struct TShaderName
{
  TShaderName() = default;

  TShaderName(const TString &Str) : Name(Str)
  {
  }

  TShaderName(const char Str[]) : Name(Str)
  {
  }

  TShaderName(const TShaderName &Str) : Name(Str.Name)
  {
  }

  TShaderName &operator=(const TString &Str)
  {
    Name = Str;
    return *this;
  }

  operator TString()
  {
    return Name;
  }

  TString Name;
};

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

  static auto GetShaderLocalPathWith(const TShaderName &Name)
  {
    return RootDirPath.string() + ShadersDir.string() + Name.Name;
  }

  static auto GetResourceDirectoryWith(const TPath &Path)
  {
    return RootDirPath.string() + ResourceDirectory.string() + Path.string();
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

  static inline TPath ShadersDir = "\\Externals\\Shaders\\";
  static inline TPath ResourceDirectory = "\\Externals\\Resources\\";

  static inline TShaderName SimpleCubeShader = "\\SimpleCube.shader";
  static inline TShaderName SimpleTextureShader = "\\SimpeTexture.shader";
};