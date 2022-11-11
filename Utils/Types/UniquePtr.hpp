#pragma once

#include <memory>

template <typename T>
using TTUniquePtr = std::unique_ptr<T>;

template <typename T, typename... Args>
TTUniquePtr<T> TTMakeUnique(Args&&... args)
{
    return std::make_unique<T>(std::forward<Args>(args)...);
}
