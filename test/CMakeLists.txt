cmake_minimum_required(VERSION 3.11)

# Test code
set(name melgan-test)
set(sources test_layers.cpp test_load.cpp test_melgan.cpp)
add_executable(${name} ${sources})

# Link
target_link_libraries(${name} PUBLIC gtest_main melgan)

# Register
add_test(NAME ${name} COMMAND ${name})
