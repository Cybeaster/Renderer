#include <cstdint>
#include <string>
#include <memory>

namespace RenderAPI
{
  class Renderer;
}

class Application
{
public:
  Application() = default;
  static std::unique_ptr<Application> application;

public:
  static  auto GetApplication()
  {
    if (!application)
        return std::move(application = std::make_unique<Application>());
  	return std::move(application);
  }

  /**
   * @brief Programm start.
   * @details Initializes Renderer class.
   *
   */
  void Start(int argc, char **argv);
};