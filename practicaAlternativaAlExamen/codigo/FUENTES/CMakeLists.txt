cmake_minimum_required(VERSION 3.16)

# Establecer la variable de compilación CMAKE_CXX_FLAGS con la opción de optimización -O2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Nombre del proyecto
project(practicaAlt_MH)

# Archivos de cabecera
include_directories(INCLUDE)

# Lista de archivos fuente
set(SOURCE_FILES
        SRC/util.cpp
        SRC/pAlternativa.cpp
        SRC/main.cpp)

# Ejecutable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Carpeta de destino para el ejecutable
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../BIN
)
