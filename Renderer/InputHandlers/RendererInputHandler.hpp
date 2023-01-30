#include "../../Utils/Types/Math.hpp"
#include "InputHandler.hpp"
#include "KeyboardKeys.hpp"
namespace RenderAPI
{

    class TRendererInputHandler
    {

    public:
        void OnWKeyToggled(bool Pressed);
        void OnSKeyToggled(bool Pressed);
        void OnDKeyToggled(bool Pressed);
        void OnAKeyToggled(bool Pressed);

        TRendererInputHandler(TVec3 &Camera, TInputHandler *Handler) : CameraRef(MakeShared(&Camera)), InputHandler(MakeShared(Handler))
        {
            InputHandler->AddListener<TRendererInputHandler, bool>(this, &TRendererInputHandler::OnWKeyToggled, EKeys::KEY_W);
            InputHandler->AddListener<TRendererInputHandler, bool>(this, &TRendererInputHandler::OnAKeyToggled, EKeys::KEY_A);
            InputHandler->AddListener<TRendererInputHandler, bool>(this, &TRendererInputHandler::OnDKeyToggled, EKeys::KEY_D);
            InputHandler->AddListener<TRendererInputHandler, bool>(this, &TRendererInputHandler::OnSKeyToggled, EKeys::KEY_S);
        }

        TRendererInputHandler() = default;

        void SetHandler(TVec3 *Camera)
        {
            CameraRef = MakeShared(Camera);
        }

    private:
        TTSharedPtr<TVec3> CameraRef;
        TTSharedPtr<TInputHandler> InputHandler;
    };

} // namespace RenderAPI
