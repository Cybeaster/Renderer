#pragma once

#include "Math.hpp"
#include "Types.hpp"

#include <format>
#include <gtx/string_cast.hpp>
#include <iostream>
#include <ostream>

#define RAPI_LOG(LogType, String, ...) \
	ODebugUtils::Log(ODebugUtils::Format(String, __VA_ARGS__), ELogType::LogType);

#define TEXT(Arg) \
	L##Arg
#define TO_C_STRING(Value) \
	ODebugUtils::ToCString(Value)

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

	FORCEINLINE static CCharPTR ToCString(const OString& String)
	{
		return String.c_str();
	}

	FORCEINLINE static CCharPTR ToCString(CCharPTR String)
	{
		return String;
	}

	template<typename... ArgTypes>
	static void Printf(const OString& Str, ArgTypes&&... Args) noexcept
	{
		std::printf(Str.c_str(), ToCString(Args)...);
	}

	template<typename... ArgTypes>
	static OString Format(std::string_view Str, ArgTypes&&... Args) noexcept
	{
		try
		{
			return std::vformat(Str, std::make_format_args(Args...));
		}
		catch (std::format_error Error)
		{
			return Error.what() + OString(Str);
		}
	}

	static OString Format(OString Str) noexcept
	{
		return Str;
	}

	static OString Format(CCharPTR Str) noexcept
	{
		return OString(Str);
	}

	static OString ToString(bool Value) noexcept
	{
		return Value ? "True" : "False";
	}

	static OString ToString(const OVec3& Vector) noexcept
	{
		return glm::to_string(Vector);
	}
};
