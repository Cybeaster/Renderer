#pragma once
#include <Types/Types.hpp>
#include <Hash.hpp>
#include "Buffer.hpp"
namespace RenderAPI
{

    struct TVertexContext
    {
        inline void Bind() const
        {
            BoundBuffer.Bind();
        }

        TVertexContext(const TVertexContext &Context) : BoundBuffer(Context.BoundBuffer),
                                                        VertexIndex(Context.VertexIndex),
                                                        VertexSize(Context.VertexSize),
                                                        IsNormalized(Context.IsNormalized),
                                                        VertexStride(Context.VertexStride)
        {
        }

        TVertexContext() = default;

        TVertexContext(const TBuffer &Buffer,
                       const uint32 Index,
                       const uint32 Size,
                       const uint32 Type,
                       const bool Normalized,
                       const uint32 Stride) : BoundBuffer(Buffer),
                                              VertexIndex(Index),
                                              VertexSize(Size),
                                              IsNormalized(Normalized),
                                              VertexStride(Stride)
        {
        }

        TVertexContext &operator=(const TVertexContext &Elem)
        {
            BoundBuffer = Elem.BoundBuffer;
            VertexIndex = Elem.VertexIndex;
            VertexSize = Elem.VertexSize;
            IsNormalized = Elem.IsNormalized;
            VertexStride = Elem.VertexStride;
            return *this;
        }

        // vertex options
        TBuffer BoundBuffer;
        uint32 VertexIndex;
        uint32 VertexSize;
        uint32 VertexType;
        uint32 VertexStride;
        bool IsNormalized;
    };

    struct TDrawContext
    {

        TDrawContext(const TDrawContext &Context) :

                                                    DrawType(Context.DrawType),
                                                    FirstDrawIndex(Context.FirstDrawIndex),
                                                    DrawSize(Context.DrawSize),
                                                    DepthFunction(Context.DepthFunction),
                                                    FrontFace(Context.FrontFace),
                                                    Flag(Context.Flag),
                                                    AttributeArray(Context.AttributeArray)
        {
        }

        TDrawContext() = default;

        TDrawContext(const uint32 Type,
                     const uint32 Index,
                     const uint32 Size,
                     const uint32 Function,
                     const uint32 FrontFaceArg,
                     const uint32 FlagArg,
                     const uint32 Array) :

                                           DrawType(Type),
                                           FirstDrawIndex(Index),
                                           DrawSize(Size),
                                           DepthFunction(Function),
                                           FrontFace(FrontFaceArg),
                                           Flag(FlagArg),
                                           AttributeArray(Array)
        {
        }

        uint32 DrawType;
        uint32 FirstDrawIndex;
        uint32 DrawSize;
        uint32 DepthFunction;
        uint32 FrontFace;
        uint32 Flag;
        uint32 AttributeArray;
    };

    class TVertexArrayHandle
    {
    public:
        struct THandleHash
        {
            auto operator()(const TVertexArrayHandle &FHandle) const
            {
                return GetHash(FHandle.Handle);
            }
        };

        TVertexArrayHandle(uint64 ID) : Handle(ID)
        {
        }

        TVertexArrayHandle(const TVertexArrayHandle &ID) : Handle(ID.Handle)
        {
        }

        TVertexArrayHandle() = default;

        uint64 GetHandle() const
        {
            return Handle;
        }

        operator int64()
        {
            return Handle;
        }

        bool operator==(const TVertexArrayHandle &ArrayHandle)
        {
            return Handle == ArrayHandle.Handle;
        }

        bool operator!=(const TVertexArrayHandle &ArrayHandle)
        {
            return Handle != ArrayHandle.Handle;
        }

        bool operator>(const TVertexArrayHandle &ArrayHandle)
        {
            return Handle > ArrayHandle.Handle;
        }

        bool operator>=(const TVertexArrayHandle &ArrayHandle)
        {
            return Handle >= ArrayHandle.Handle;
        }

        bool operator<(const TVertexArrayHandle &ArrayHandle)
        {
            return Handle < ArrayHandle.Handle;
        }

        bool operator<=(const TVertexArrayHandle &ArrayHandle)
        {
            return Handle <= ArrayHandle.Handle;
        }

        TVertexArrayHandle &operator=(const TVertexArrayHandle &OtherHandle)
        {
            Handle = OtherHandle.Handle;
            return *this;
        }

        friend bool operator==(const TVertexArrayHandle &FirstHandle, const TVertexArrayHandle &SecondHandle)
        {
            return FirstHandle.Handle == SecondHandle.Handle;
        }

    private:
        uint64 Handle;
    };

    class TVertexArrayElem
    {

    public:
        TVertexArrayElem(const TVertexContext &Vertex, const TDrawContext &Draw) noexcept
            : VertexContext(Vertex),
              DrawContext(Draw)
        {
        }

        TVertexArrayElem(const TVertexArrayElem &Elem) noexcept
            : VertexContext(Elem.VertexContext),
              DrawContext(Elem.DrawContext)
        {
        }

        TVertexArrayElem() = default;
        ~TVertexArrayElem() noexcept
        {}

        TVertexArrayElem &operator=(const TVertexArrayElem &Elem)
        {
            VertexContext = Elem.VertexContext;
            DrawContext = Elem.DrawContext;
            return *this;
        }

        void DrawBuffer() const;

    private:
        TVertexContext VertexContext;
        TDrawContext DrawContext;

        // draw options
    };
} // namespace RenderAPI
