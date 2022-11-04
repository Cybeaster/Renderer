#pragma once
#include "../Types.hpp"

namespace RenderAPI
{
    template <typename Type, uint32 Index, uint32 Size>
    struct TTupleElem
    {
        template <typename ArgType>
        TTupleElem(ArgType &&Arg) : Value(std::forward<ArgType>(Arg)) {}
        TTupleElem() : Value() {}

        ~TTupleElem() {}

        Type Value;
    };

    template <typename Type>
    struct TTupleElem<Type, 0, 2>
    {
        template <typename ArgType>
        TTupleElem(ArgType &&Arg) : TargetValue(std::forward<ArgType>(Arg)) {}
        TTupleElem() : TargetValue() {}

        ~TTupleElem() {}

        Type TargetValue;
    };

    template <uint32 Index, uint32 TupleSize>
    struct TTupleElemGetterByIndex
    {
        template <typename DeducedType, typename TupleType>
        static decltype(auto) GetImpl(const TTupleElem<DeducedType, Index, TupleSize> &, TupleType &&Tuple)
        {
            return std::forward<TTupleElem<DeducedType, Index, TupleSize>>(Tuple).Value;
        }

        template <typename TupleType>
        static decltype(auto) Get(TupleType &&Tuple)
        {
            return GetImpl(Tuple, std::forward<TupleType>(Tuple));
        }
    };

    template <>
    struct TTupleElemGetterByIndex<0, 2>
    {
        template <typename Type>
        static auto Get(Type &&Tuple)
        {
            return static_cast<TTupleElem<decltype(Tuple.TargetValue), 0, 2>>(Tuple).TargetValue;
        }
    };


    template<typename Type, uint32 TupleSize>
    struct TTupleElemGetterByType
    {

        template<uint32 DeducedIndex, typename TupleType>
        static decltype(auto) GetImpl(const TTupleElem<Type,DeducedIndex,TupleSize>&, TupleType&& Tuple)
        {
            return TTupleElemGetterByIndex<DeducedIndex,TupleSize>::Get(std::forward<TupleType>(Tuple));
        }


        template<typename TupleType>
        static decltype(auto) Get(TupleType&& Tuple)
        {
            return GetImpl(Tuple,std::forward<TupleType>(Tuple));
        }
    };
    

} // namespace RenderAPI
