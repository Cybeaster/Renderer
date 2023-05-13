#pragma once
#include "Logging/Printer.hpp"

class IPrintable
{
public:
	virtual void Print(OPrinter* Printer) = 0;
};