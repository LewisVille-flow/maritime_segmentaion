#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(RealSenseImShowExample)

add_executable(rs-imshow rs-imshow.cpp)
add_executable(rs-record rs-record.cpp)

set_property(TARGET rs-imshow rs-record PROPERTY CXX_STANDARD 11)
target_link_libraries(rs-imshow ${DEPENDENCIES})
target_link_libraries(rs-record ${DEPENDENCIES})
set_target_properties (rs-imshow rs-record PROPERTIES
	FOLDER "Examples/OpenCV"
)

install(
	TARGETS

	rs-imshow

	RUNTIME DESTINATION
	${CMAKE_INSTALL_PREFIX}/bin
)
