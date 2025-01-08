set(LLVM_LINK_COMPONENTS support)

add_clang_executable(false-sharing-detect
  FalseSharing.cpp
  )
target_link_libraries(false-sharing-detect
  PRIVATE
  clangTooling
  clangBasic
  clangASTMatchers
  )
