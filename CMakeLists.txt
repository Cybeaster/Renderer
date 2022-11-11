cmake_minimum_required(VERSION 3.21)

project(OpenGLGraphics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

find_package(OpenGL REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/Externals/Include/glfw
	${PROJECT_SOURCE_DIR}/Externals/Include/glm
	${PROJECT_SOURCE_DIR}/Externals/Include/glew
	${PROJECT_SOURCE_DIR}/Externals/Include/SOIL2
	${PROJECT_SOURCE_DIR}/Externals/Include/cuda
	${PROJECT_SOURCE_DIR}/Renderer
	${PROJECT_SOURCE_DIR}/Renderer/Removed
	${PROJECT_SOURCE_DIR}/Shader
	${PROJECT_SOURCE_DIR}/Texture
	${PROJECT_SOURCE_DIR}/Externals/Vendor
	${PROJECT_SOURCE_DIR}/Test/Particle
	${PROJECT_SOURCE_DIR}/Test
	${PROJECT_SOURCE_DIR}/Test/Tube
	${PROJECT_SOURCE_DIR}/Test/Tube/Plane
	${PROJECT_SOURCE_DIR}/Utils
	${PROJECT_SOURCE_DIR}/Externals/freeglut/include
	${PROJECT_SOURCE_DIR}/Utils/Threads/ThreadPool
	${PROJECT_SOURCE_DIR}/Threads
	${PROJECT_SOURCE_DIR}/Utils/Threads/HierarchicalMutex
	${PROJECT_SOURCE_DIR}/Utils/Checks
	${PROJECT_SOURCE_DIR}/Utils/Types
	${PROJECT_SOURCE_DIR}/Utils/Types/Tuple

)

set(GLFW_LIB ${PROJECT_SOURCE_DIR}/Externals/Lib/glfw3.lib)
set(GLEW_LIB ${PROJECT_SOURCE_DIR}/Externals/Lib/glew32.lib)
set(SOIL_LIB ${PROJECT_SOURCE_DIR}/Externals/Lib/soil2-debug.lib)

add_executable(exec
	main.cpp
	Renderer/Renderer.cpp
	Renderer/VertexBuffer.cpp
	Application/Application.cpp
	Shader/Shader.cpp
	Texture/Texture.cpp
	Test/Test.cpp
	Test/TestTexture.cpp
	Test/TestSimpleCube.cpp
	Test/Particle/Particle.cpp
	Test/SimpleBox.cpp
	Test/Particle/TestParticles.cpp
	Utils/Threads/ThreadPool/ThreadPool.cpp
	Utils/Threads/HierarchicalMutex/HierarchicalMutex.cpp

	Renderer/Renderer.hpp
	Renderer/VertexBuffer.hpp
	Application/Application.hpp
	Shader/Shader.hpp
	Texture/Texture.hpp
	Test/Test.hpp
	Test/TestTexture.hpp
	Test/TestSimpleCube.hpp
	Test/Particle/Particle.hpp
	Test/Particle/ElectroField.hpp
	Test/SimpleBox.hpp
	Test/Particle/TestParticles.hpp
	Utils/Threads/ThreadPool/ThreadPool.hpp
	Utils/Threads/HierarchicalMutex/HierarchicalMutex.hpp
	Utils/Checks/Exception.hpp
	Utils/Checks/Assert.hpp
	Utils/Functor/Functor.hpp

	Utils/Types/Math.hpp
	Utils/Types/Queue.hpp
	Utils/Types/Thread.hpp
	Utils/Types/Types.hpp
	Utils/Types/Vector.hpp
	Utils/Types/UniquePtr.hpp
	Utils/Types/Hash.hpp
	Utils/Types/Pair.hpp
	Utils/Types/Tuple.hpp
	Utils/Types/Tuple/Tuple.hpp

)
target_link_libraries(
	exec
	PRIVATE
	${GLFW_LIB}
	${OPENGL_LIBRARY}
	${GLEW_LIB}
	${SOIL_LIB}
)
