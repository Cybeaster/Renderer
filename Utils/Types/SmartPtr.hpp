#pragma once
#include <memory>

namespace RenderAPI
{
    template <typename T>
    __forceinline auto MakeShared(T Arg)
    {
        return std::make_shared<T>;
    }

    template <typename T>
    __forceinline auto MakeUnique(T Arg)
    {
        return std::make_unique<T>;
    }

    template <typename T>
    using TTWeakPtr = std::weak_ptr<T>;

    template <typename T>
    using TTSharedPtr = std::shared_ptr<T>;

    template <typename T>
    using TTUniquePtr = std::unique_ptr<T>;
}
