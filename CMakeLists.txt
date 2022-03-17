cmake_minimum_required(VERSION 3.21)

project(Opengl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

find_package(OpenGL REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}/Externals/glfw/include
	${PROJECT_SOURCE_DIR}/Externals/glm
	${PROJECT_SOURCE_DIR}/Externals/glew/include
	${PROJECT_SOURCE_DIR}/Renderer
)


set(GLFW_LIB ${PROJECT_SOURCE_DIR}/Externals/glfw/lib-vc2019/glfw3.lib)
set(GLEW_LIB ${PROJECT_SOURCE_DIR}/Externals/glew/lib/Release/x64/glew32s.lib)


add_executable(exec 
main.cpp
Application/Application.cpp
Renderer/Renderer.cpp
Renderer/VertexBuffer.cpp
Renderer/IndexBuffer.cpp
)

target_link_libraries(exec ${GLFW_LIB} ${OPENGL_LIBRARY} ${GLEW_LIB})