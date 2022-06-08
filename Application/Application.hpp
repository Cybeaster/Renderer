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

      Application() = default;
      static Application* application;

    public:

      static Application* GetApplication()
      {
          if(application == nullptr)
          {
              application = new Application();
              return application;
          }
          else
            return application;
      }

      /**
       * @brief Вход в программу.
       * @details Инициализирует класс Renderer, отвечающий за создание окна и контекста для OpenGL.
       * 
       */
      void Start(int argc, char **argv);

};