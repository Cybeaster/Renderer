#include <cstdint>
#include <string>
#include <UniquePtr.hpp>

namespace RenderAPI
{
  class Renderer;
}

class Application
{
private:
  static TTUniquePtr<Application> application;

public:
	Application() = default;

  static auto GetApplication()
  {
    if (!application)
        return std::move(application = TTMakeUnique<Application>());
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