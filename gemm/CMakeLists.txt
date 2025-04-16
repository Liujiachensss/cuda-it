# add_executable(matmul matmul.cu)
# target_link_libraries(matmul PRIVATE cublas)

add_executable(matmuls matmuls.cu)
target_link_libraries(matmuls PRIVATE cublas)