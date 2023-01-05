#include "VertexArray.hpp"
#include "GL/glew.h"
#include "glfw3.h"
#include "Renderer.hpp"
namespace RenderAPI
{
    TVertexArray::TVertexArray(/* args */)
    {
    }

    void TVertexArray::AddVertexArray()
    {
        TVertexIndex id;
        GLCall(glGenVertexArrays(1, &id.Index));
        GLCall(glBindVertexArray(id.Index));

        VertexIndicesArray.push_back(id);
    }

    TVertexArrayHandle TVertexArray::CreateVertexElement(const TVertexContext &VContext, const TDrawContext &RContext)
    {
        ++VertexCounter;
        auto handle = TVertexArrayHandle(VertexCounter);
        VertexElements[handle] = TVertexArrayElem(VContext, RContext);
        return handle;
    }

    void TVertexArray::DrawBuffer(const TVertexArrayHandle &Handle) const
    {
        auto elem = VertexElements.find(Handle);
        elem->second.DrawBuffer();
    }

    TVertexArray::~TVertexArray()
    {
    }
} // namespace RenderAPI
