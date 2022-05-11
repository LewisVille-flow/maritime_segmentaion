cmake_minimum_required(VERSION 3.3)
file(GLOB segnetSources *.cpp)
file(GLOB segnetIncludes *.h )
find_package(CUDA)
find_package(jetson-utils)
find_package(jetson-inference)
find_package(OpenCV REQUIRED)

link_directories(/usr/lib/aarch64-linux-gnu/tegra)
cuda_add_executable(segnet ${segnetSources})
target_link_libraries(segnet jetson-inference jetson-utils ${OpenCV_LIBS})
install(TARGETS segnet DESTINATION bin)


# for legacy users of segnet-camera (will be deprecated)
#cuda_add_executable(segnet-camera ${segnetSources})
#target_link_libraries(segnet-camera jetson-inference)
#install(TARGETS segnet-camera DESTINATION bin)

# for legacy users of segnet-console (will be deprecated)
#cuda_add_executable(segnet-console ${segnetSources})
#set_target_properties(segnet-console PROPERTIES COMPILE_FLAGS "-DHEADLESS" )
#target_link_libraries(segnet-console jetson-inference)
#install(TARGETS segnet-console DESTINATION bin)
