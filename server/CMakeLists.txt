cmake_minimum_required(VERSION 3.15)
project(quantDataBase CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Drogon CONFIG REQUIRED)
find_package(libpqxx CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(${PROJECT_NAME} main.cc)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    Drogon::Drogon 
    libpqxx::pqxx 
    nlohmann_json::nlohmann_json
)