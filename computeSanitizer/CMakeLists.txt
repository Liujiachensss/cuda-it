add_executable(cs-writeValidMem writeValidMem.cu)
target_compile_options(cs-writeValidMem PRIVATE -lineinfo)
# target_compile_options(cs-writeValidMem PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)

add_executable(cs-racecheck racecheck.cu)
target_compile_options(cs-racecheck PRIVATE -lineinfo)