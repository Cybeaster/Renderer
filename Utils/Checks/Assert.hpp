#pragma once

#include "Logging/Log.hpp"
#include "Types.hpp"
#include "Types/GL.hpp"

#include <cassert>

#define ASSERT_MSG(x, ...)        \
	RAPI_LOG(Error, __VA_ARGS__); \
	assert(x);

#define ASSERT(x) \
	assert(x);

#ifdef NDEBUG
#define ENSURE(x, ...) \
	x
#else
#define ENSURE(x, ...) \
	Ensure((x), __VA_ARGS__)
#endif // NDEBUG

#ifdef NDEBUG
#define GLCall(x)
#else
#define GLCall(x)   \
	GLClearError(); \
	x;              \
	ASSERT(GLLogCall(#x, __FILE__, __LINE__))
#endif // NDEBUG

template<typename... ArgTypes>
inline bool Ensure(bool Value, ArgTypes&&... Args)
{
	if (!Value)
	{
		RAPI_LOG(Error, Args...);
		__debugbreak();
	}
	return Value;
}

inline bool Ensure(bool Value)
{
	if (!Value)
	{
		__debugbreak();
	}
	return Value;
}

inline void GLClearError()
{
	while (glGetError() != GL_NO_ERROR)
		;
}

inline bool GLLogCall(const char* func, const char* file, const int line)
{
	while (const GLenum error = glGetError())
	{
		std::cout << "[Opengl Error] (" << std::hex << error << ") :" << func << '\t' << line << '\t' << file << std::endl;
		return false;
	}
	return true;
}