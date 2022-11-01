#pragma once
#include "../Functions.hpp"
#include "TupleElem.hpp"

namespace RenderAPI
{
    template <typename FunctionType, typename... ArgTypes>
    auto CallFunction(FunctionType &&FuncType, ArgTypes &&...Args) -> decltype(std::forward<FunctionType>(FuncType)(std::forward<ArgTypes>(Args)...))
    {
        return std::forward(FuncType)(std::forward(Args)...);
    }
#pragma region TupleBase

    template <typename Indices, typename... Types>
    struct TTupleBase;

    template <uint32... Indices, typename... Types>
    struct TTupleBase<TIntegerSequenceWrapper<uint32, Indices...>, Types...> : TTupleElem<Types, Indices, sizeof...(Types)>...
    {
        class TTFunctor;
        friend TTFunctor;

        template <typename... ArgTypes>
        TTupleBase(ArgTypes &&...Args) : TTupleElem<ArgTypes, Indices, sizeof...(ArgTypes)>(std::forward<ArgTypes>(Args))...
        {
        }
        ~TTupleBase() {}

#pragma region GetByIndex
        template <uint32 Index>
        decltype(auto) Get() &&
        {
            return TTupleElemGetterByIndex<Index, sizeof...(Types)>::Get(static_cast<TTupleBase &&>(*this));
        }

        template <uint32 Index>
        decltype(auto) Get() const &
        {
            return TTupleElemGetterByIndex<Index, sizeof...(Types)>::Get(static_cast<const TTupleBase &>(*this));
        }

        template <uint32 Index>
        decltype(auto) Get() &
        {
            return TTupleElemGetterByIndex<Index, sizeof...(Types)>::Get(static_cast<TTupleBase &>(*this));
        }

        template <uint32 Index>
        decltype(auto) Get() const &&
        {
            return TTupleElemGetterByIndex<Index, sizeof...(Types)>::Get(static_cast<const TTupleBase &&>(*this));
        }
#pragma endregion GetByIndex
#pragma region GetByType
        template <typename T>
        decltype(auto) Get() &
        {
            return TTupleElemGetterByType<T, sizeof...(Types)>::Get(static_cast<TTupleBase &>(*this));
        }
        template <typename T>
        decltype(auto) Get() &&
        {
            return TTupleElemGetterByType<T, sizeof...(Types)>::Get(static_cast<TTupleBase &&>(*this));
        }
        template <typename T>
        decltype(auto) Get() const &
        {
            return TTupleElemGetterByType<T, sizeof...(Types)>::Get(static_cast<const TTupleBase &>(*this));
        }
        template <typename T>
        decltype(auto) Get() const &&
        {
            return TTupleElemGetterByType<T, sizeof...(Types)>::Get(static_cast<const TTupleBase &&>(*this));
        }
#pragma endregion GetByType

        template <typename FuncType>
        decltype(auto) Call(FuncType &&Function) &&
        {
            ::Execute(Function, static_cast<TTupleBase &&>(*this).Get<Indices>()...);
        }

        template <typename FuncType>
        decltype(auto) Call(FuncType &&Function) &
        {
            ::Execute(Function, static_cast<TTupleBase &>(*this).Get<Indices>()...);
        }

        template <typename FuncType>
        decltype(auto) Call(FuncType &&Function) const &&
        {
            ::Execute(Function, static_cast<const TTupleBase &&>(*this).Get<Indices>()...);
        }

        template <typename FuncType>
        decltype(auto) Call(FuncType &&Function) const &
        {
            return ::Execute(Function, static_cast<const TTupleBase &>(*this) Get<Indices>()...);
        }
    };

#pragma endregion TupleBase

#pragma region Tuple
    template <typename... Types>
    class TTElemSequenceTuple;

    template <typename... Types>
    class TTElemSequenceTuple : public TTupleBase<TTMakeIntegerSequence<uint32, sizeof...(Types)>, Types...>
    {
        using IntegerSequence = TTMakeIntegerSequence<uint32, sizeof...(Types)>;
        using Super = TTupleBase<IntegerSequence, Types...>;

    public:
        template <typename... ArgTypes>
        TTElemSequenceTuple(ArgTypes &&...Args) : Super(std::forward<ArgTypes>(Args)...)
        {
        }
        ~TTElemSequenceTuple() {}
    };

#pragma endregion Tuple

} // namespace RenderAPI
