cmake_minimum_required(VERSION 3.21)

project(Opengl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

find_package(OpenGL REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}/Externals/Include/glfw
	${PROJECT_SOURCE_DIR}/Externals/Include/glm
	${PROJECT_SOURCE_DIR}/Externals/Include/glew
	${PROJECT_SOURCE_DIR}/Externals/Include/SOIL2
	${PROJECT_SOURCE_DIR}/Renderer
	${PROJECT_SOURCE_DIR}/VertexArray
	${PROJECT_SOURCE_DIR}/Shader
	${PROJECT_SOURCE_DIR}/Texture
	${PROJECT_SOURCE_DIR}/Externals/Vendor
	${PROJECT_SOURCE_DIR}/Test
		)


set(GLFW_LIB ${PROJECT_SOURCE_DIR}/Externals/Lib/glfw3.lib)
set(GLEW_LIB ${PROJECT_SOURCE_DIR}/Externals/Lib/glew32s.lib)
set(SOIL_LIB ${PROJECT_SOURCE_DIR}/Externals/Lib/soil2-debug.lib)


add_executable(exec 
main.cpp
Application/Application.cpp
VertexArray/VertexArray.cpp
Shader/Shader.cpp
Texture/Texture.cpp
Renderer/Renderer.cpp
Test/TestClearColor.cpp
Test/Test.cpp
Test/TestTexture.cpp
Test/TestSimpleCube.cpp
Test/Particle/TestParticles.cpp
Test/Particle/Particle.cpp
)

target_link_libraries(exec ${GLFW_LIB} ${OPENGL_LIBRARY} ${GLEW_LIB} ${SOIL_LIB})