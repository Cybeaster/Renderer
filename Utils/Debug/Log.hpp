#pragma once

#include "Types.hpp"

#include <format>
#include <iostream>
#include <ostream>

#define RAPI_LOG(LogType, String, ...) \
	ODebugUtils::Log(std::printf(String, __VA_ARGS__), ELogType::LogType);

#define TO_STRING(Value) \
	ODebugUtils::ToString(Value)

enum class ELogType
{
	Log,
	Warning,
	Error,
	Critical
};

class ODebugUtils
{
public:
	static inline std::ostream& DebugOutput = std::cout;

	template<typename Object>
	FORCEINLINE static void Log(Object&& String, ELogType Type = ELogType::Log) noexcept
	{
		switch (Type)
		{
		case ELogType::Log:
			DebugOutput << "\n"
			            << "Log: \t" << String << std::endl;
			break;

		case ELogType::Warning:
			DebugOutput << "\n"
			            << "Warning: \t" << String << std::endl;
			break;

		case ELogType::Error:
			DebugOutput << "\n \t \t"
			            << "Error: \t" << String << std::endl;
			break;

		case ELogType::Critical:
			DebugOutput << "\n \t \t"
			            << "Critical: \t" << String << std::endl;
			break;
		}
	}
	
	FORCEINLINE static OString ToString(bool Value) noexcept
	{
		return Value ? "True" : "False";
	}
};