set(LLVM_LINK_COMPONENTS support)

add_clang_executable(get-func-names
  GetFuncNames.cpp
  )
target_link_libraries(get-func-names
  PRIVATE
  clangTooling
  clangBasic
  clangASTMatchers
  )
