cmake_minimum_required(VERSION 3.18)
project(rpi_drivers_cpp)

set(SOURCE_FILES ../src/main.cpp 
                 ../src/led.cpp 
                 ../src/button.cpp 
                 ../src/gpiod_utils.c)

include_directories(../inc)
add_executable(rpi_drivers_cpp_example ${SOURCE_FILES})
target_compile_options(rpi_drivers_cpp_example PRIVATE -Wall -Werror)
target_link_libraries(rpi_drivers_cpp_example gpiod)
set_target_properties(rpi_drivers_cpp_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../out)
set_property(TARGET rpi_drivers_cpp_example PROPERTY CXX_STANDARD 17)