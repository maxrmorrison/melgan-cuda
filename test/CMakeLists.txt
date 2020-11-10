set(sources
    test.cpp
)

set(name test)

add_executable(name ${sources})
target_link_libraries(name PUBLIC
    gtest_main
    melgan
)

add_test(
    NAME name
    COMMAND test
)
