cmake_minimum_required(VERSION 3.20)
project(s1_gguf LANGUAGES CXX)

set(LLAMA_CPP_DIR "" CACHE PATH "Pinned llama.cpp checkout with built shared libraries")
set(LLAMA_CPP_REVISION "5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd")
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_DIR}" rev-parse HEAD
                OUTPUT_VARIABLE actual_revision OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE revision_result)
if(NOT revision_result EQUAL 0 OR NOT actual_revision STREQUAL LLAMA_CPP_REVISION)
  message(FATAL_ERROR "LLAMA_CPP_DIR must point to the pinned ${LLAMA_CPP_REVISION} checkout")
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" -c core.fsmonitor=false -C "${LLAMA_CPP_DIR}" diff --quiet HEAD --
                RESULT_VARIABLE dirty_result)
if(NOT dirty_result EQUAL 0)
  message(FATAL_ERROR "Pinned llama.cpp tracked source must be pristine before building the adapter")
endif()
foreach(lib llama mtmd ggml ggml-base)
  string(REPLACE "-" "_" variable "${lib}")
  find_library(${variable}_LIBRARY NAMES ${lib}
    PATHS "${LLAMA_CPP_DIR}/build/bin" NO_DEFAULT_PATH REQUIRED)
  list(APPEND native_libraries "${${variable}_LIBRARY}")
endforeach()
add_executable(s1-gguf s1_gguf.cpp)
target_compile_features(s1-gguf PRIVATE cxx_std_17)
target_compile_definitions(s1-gguf PRIVATE S1_LLAMA_CPP_REVISION="${LLAMA_CPP_REVISION}")
target_include_directories(s1-gguf PRIVATE
  "${LLAMA_CPP_DIR}/include" "${LLAMA_CPP_DIR}/ggml/include"
  "${LLAMA_CPP_DIR}/tools/mtmd" "${LLAMA_CPP_DIR}/vendor")
target_link_libraries(s1-gguf PRIVATE ${native_libraries})
set_target_properties(s1-gguf PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  BUILD_RPATH "${LLAMA_CPP_DIR}/build/bin")
