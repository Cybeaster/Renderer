cmake_minimum_required(VERSION 3.21)

project(OpenGLGraphics LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED)

set(USE_CUDA OFF)

add_library(
	    cudaTarget 
		Test/Particle/TestParticles.cu
		Test/Tube/TestTube.cu)

if(OFF)
	find_package(CUDAToolkit REQUIRED)
	
	set_target_properties(cudaTarget
	PROPERTIES  
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	
endif(OFF)

find_package(OpenGL REQUIRED)


include_directories(
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
Test/Tube/Tube.cpp


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
Test/Particle/StaticParticle.hpp
Test/Tube/TestTube.hpp
Test/Tube/Tube.hpp



Test/Tube/Plane/Line.cpp
Test/Tube/Plane/Line.h
Test/Tube/Plane/mainTest.cpp
Test/Tube/Plane/Pipe.cpp
Test/Tube/Plane/Pipe.h
Test/Tube/Plane/Plane.cpp
Test/Tube/Plane/Plane.h
Test/Tube/Plane/Timer.cpp
Test/Tube/Plane/Timer.h
Test/Tube/Plane/Vectors.h
Test/Tube/Plane/Matrices.h
Test/Tube/Plane/Matrices.cpp
)


	target_link_libraries(
		exec
		PRIVATE
		${GLFW_LIB} 
		${OPENGL_LIBRARY}
		${GLEW_LIB} 
		${SOIL_LIB}
		cudaTarget)


