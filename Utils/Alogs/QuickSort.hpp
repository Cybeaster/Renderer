//
// Created by Cybea on 3/8/2023.
//

#ifndef RENDERAPI_QUICKSORT_HPP
#define RENDERAPI_QUICKSORT_HPP

#include "List.hpp"
#include "Thread.hpp"

#include <algorithm>
namespace Algo
{
template<typename Container>
Container SequentialQuickSort(Container Input);

template<typename T>
OTList<T> SequentialQuickSort(OTList<T> Input)
{
	if (Input.empty())
	{
		return Input;
	}
	OTList<T> output;

	output.splice(output.begin(),Input,Input.end());

	T const& pivot = *output.begin();

	auto dividePoint = std::partition(Input.begin(),Input.end(),[&](T const& Value){
		                                  return Value < pivot;
	});

	OTList<T> lowerPart;
	lowerPart.splice(lowerPart.end(),Input,Input.begin(),dividePoint);

	auto newLower(SequentialQuickSort(Move(lowerPart)));
	auto newHigher(SequentialQuickSort(Move(Input)));

	output.splice(output.end(),newHigher);
	output.splice(output.begin(),newLower);

	return output;
}

template <typename T>

OTList<T> AsyncQuickSort(OTList<T> Input)
{
	if (Input.empty())
	{
		return Input;
	}
	OTList<T> output;

	output.splice(output.begin(),Input,Input.end());

	T const& pivot = *output.begin();

	auto dividePoint = std::partition(Input.begin(),Input.end(),[&](T const& Value){
		                                  return Value < pivot;
	                                  });

	OTList<T> lowerPart;
	lowerPart.splice(lowerPart.end(),Input,Input.begin(),dividePoint);

	OFuture<OTList<T>> newLower(std::async(AsyncQuickSort(Move(lowerPart))));

	auto newHigher(AsyncQuickSort(Move(Input)));

	output.splice(output.end(),newHigher);
	output.splice(output.begin(),newLower.get());

	return output;
}

} // namespace Algo

#endif // RENDERAPI_QUICKSORT_HPP
