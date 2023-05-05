#pragma once

#ifndef RENDERAPI_PRINTER_HPP
#define RENDERAPI_PRINTER_HPP
#include "Types.hpp"

#include <cstdio>
#include <iostream>
class OPrinter
{

public:

	template<class Printable>
	FORCEINLINE OPrinter& operator<<(Printable Object)
	{
		std::cout << Object;
		return *this;
	}

	template<class Printable>
	FORCEINLINE OPrinter& Print(Printable Object)
	{
		std::printf(Object);
		return *this;
	}
};
#endif // RENDERAPI_PRINTER_HPP
