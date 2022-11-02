#pragma once

template <typename FunctionType, typename... ArgTypes>
auto Execute(FunctionType &&FuncType, ArgTypes &&...Args) -> decltype(std::forward<FunctionType>(FuncType)(std::forward<ArgTypes>(Args)...))
{
    return std::forward<FunctionType>(FuncType)(std::forward<ArgTypes>(Args)...);
}