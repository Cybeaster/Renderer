#include <cstdint>
#include <string>
#include "SmartPtr.hpp"

namespace RenderAPI
{
  class Renderer;
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

  /**
   * @brief Programm start.
   * @details Initializes Renderer class.
   *
   */
  void Start(int argc, char **argv);

private:
  Application() = default;

  static inline RenderAPI::TTSharedPtr<Application> application = nullptr;
};