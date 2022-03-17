#include <cstdint>
#include <string>


struct ShaderSource
{
    std::string vertexShader;
    std::string fragmentShader;
};
enum class ShaderType
{
    NONE = -1,
    VERTEX = 0,
    FRAGMENT = 1
};

class Application 
{
    public:

    void Start();
   

};