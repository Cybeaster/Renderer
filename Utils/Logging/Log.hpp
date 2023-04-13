#pragma once

#include "Math.hpp"
#include "Printer.hpp"
#include "Types.hpp"

#include <format>
#include <gtx/string_cast.hpp>
#include <iostream>
#include <ostream>
#include <sstream>

#define RAPI_LOG(LogType, String, ...) \
	SLogUtils::Log(SLogUtils::Format(String, __VA_ARGS__), ELogType::LogType);

#define TEXT(Arg) \
	L##Arg
#define TO_C_STRING(Value) \
	ODebugUtils::ToCString(Value)

#define TO_STRING(Value) \
	SLogUtils::ToString(Value)

enum class ELogType
{
	Log,
	Warning,
	Error,
	Critical
};

struct SLogUtils
{
public:
	inline static OPrinter Printer;

	template<typename Object>
	FORCEINLINE static void Log(const Object& String, ELogType Type = ELogType::Log) noexcept
	{
		switch (Type)
		{
		case ELogType::Log:
			std::cout << "\n"
			          << "Log: \t" << String << std::endl;
			break;

		case ELogType::Warning:
			std::cout << "\n"
			          << "Warning: \t" << String << std::endl;
			break;

		case ELogType::Error:
			std::cout << "\n \t \t"
			          << "Error: \t" << String << std::endl;
			break;

		case ELogType::Critical:
			std::cout << "\n \t \t"
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
	static OString Format(std::wstring_view Str, ArgTypes&&... Args)
	{
		try
		{
			return std::vformat(Str, std::make_format_args(Args...));
		}
		catch (const std::format_error& error)
		{
			return error.what() + OString(Str.begin(), Str.end());
		}
	}

	template<typename... ArgTypes>
	static OString Format(std::string_view Str, ArgTypes&&... Args)
	{
		try
		{
			return std::vformat(Str, std::make_format_args(Args...));
		}
		catch (const std::format_error& error)
		{
			return error.what() + OString(Str);
		}
	}

	static OString Format(OString Str)
	{
		return Str;
	}

	static OString Format(CCharPTR Str)
	{
		return { Str };
	}

	template<typename T>
	static OString ToString(T Value)
	{
		return std::to_string(Value);
	}

	template<>
	static OString ToString(bool Value)
	{
		return Value ? "True" : "False";
	}

	template<>
	static OString ToString(OVec3 Vector)
	{
		return glm::to_string(Vector);
	}

	template<template<typename Contained, typename Alloc = std::allocator<Contained>>
	         typename Container,
	         typename Type>
	static OString ToString(const Container<Type>& Array) noexcept
	{
		std::stringstream ss;
		for (const auto& elem : Array)
		{
			ss << elem;
		}
		return ss.str();
	}
};
