get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries({vcpk_repository}/scripts/buildsystems/vcpkg.cmake)