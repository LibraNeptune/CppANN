cmake_minimum_required(VERSION 3.15)
project(rnn)
include_directories(C:/Users/zsms1/CLionProjects/adept-2.0.5/include)
include_directories(C:/Users/zsms1/AppData/Local/Programs/Python/Python37/include)
set(CMAKE_CXX_STANDARD 14)

add_executable(rnn main.cpp basic_class.h basic_class.cpp)
target_link_libraries(rnn C:/Users/zsms1/CLionProjects/OpenBLAS-0.3.6-x64/lib/libopenblas.a)
target_link_libraries(rnn C:/Users/zsms1/AppData/Local/Programs/Python/Python37/libs/libpython37.a)