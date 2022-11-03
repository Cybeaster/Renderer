#include <cstdint>
#include <string>
#include <memory>

namespace RenderAPI
{
  class Renderer;
}

class Application
{
private:
  static std::unique_ptr<Application> application;

public:
	Application() = default;

  static auto GetApplication()
  {
    if (!application)
        return std::move(application = std::make_unique<Application>());
    else
  		return std::move(application);
  }

  /**
   * @brief Programm start.
   * @details Initializes Renderer class.
   *
   */
  void Start(int argc, char **argv);
};