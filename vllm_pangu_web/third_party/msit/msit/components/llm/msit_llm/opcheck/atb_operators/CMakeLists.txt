cmake_minimum_required(VERSION 3.15)
set(LIB_NAME opchecker)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_SKIP_RPATH TRUE)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/dependency
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include
    $ENV{ASCEND_TOOLKIT_HOME}/x86_64-linux/include
)

link_directories(
    $ENV{ATB_SPEED_HOME_PATH}/lib
    $ENV{ATB_HOME_PATH}/lib
)
link_libraries(atb atb_train atb_speed_core)
add_library(${LIB_NAME} SHARED ${CMAKE_CURRENT_LIST_DIR}/operation_creator.cpp)

target_compile_options(${LIB_NAME} PRIVATE
    -fPIC
    -pie
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2
    -ftrapv
)
add_compile_options(-fstack-check)
target_link_options(${LIB_NAME} PRIVATE
    -Wl,-z,relro,-z,now,-z,noexecstack
    -s
)

message("-- CMAKE_CXX_FLAGS is ${CMAKE_CXX_FLAGS}")
message("-- ATB_VERSION is ${ATB_VERSION}")
add_compile_options(-DATB_VERSION=${ATB_VERSION})

install(
    TARGETS ${LIB_NAME}
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)
