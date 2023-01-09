#pragma once
#include "Types.hpp"

namespace RenderAPI
{

    struct TDrawContext
    {

        TDrawContext(const TDrawContext &Context) :

                                                    DrawType(Context.DrawType),
                                                    FirstDrawIndex(Context.FirstDrawIndex),
                                                    DrawSize(Context.DrawSize),
                                                    DepthFunction(Context.DepthFunction),
                                                    FrontFace(Context.FrontFace),
                                                    Flag(Context.Flag)
        {
        }

        TDrawContext() = default;

        TDrawContext(const uint32 Type,
                     const uint32 Index,
                     const uint32 Size,
                     const uint32 Function,
                     const uint32 FrontFaceArg,
                     const uint32 FlagArg) :

                                           DrawType(Type),
                                           FirstDrawIndex(Index),
                                           DrawSize(Size),
                                           DepthFunction(Function),
                                           FrontFace(FrontFaceArg),
                                           Flag(FlagArg)
        {
        }

        uint32 DrawType;
        uint32 FirstDrawIndex;
        uint32 DrawSize;
        uint32 DepthFunction;
        uint32 FrontFace;
        uint32 Flag;
    };
} // namespace RenderAPI
