cmake_minimum_required(VERSION 3.8)
project(legged_remap_rl_controller)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(onnxruntime REQUIRED)
find_package(Eigen3 REQUIRED)

ament_auto_add_library(
        ${PROJECT_NAME}
        SHARED
        DIRECTORY src
)

# prevent pluginlib from using boost
target_compile_definitions(${PROJECT_NAME} PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")
pluginlib_export_plugin_description_file(controller_interface legged_remap_rl_controller.xml)

target_include_directories(
        ${PROJECT_NAME} PUBLIC
        ${onnxruntime_INCLUDE_DIRS} 
        ${EIGEN3_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
        ${onnxruntime_LIBRARIES}  
        Eigen3::Eigen)

ament_auto_package()
