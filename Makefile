define newline


endef

$(error Build system changed:$(newline)\
The Makefile build has been replaced by CMake.$(newline)$(newline)\
For BeeLlama build instructions see:$(newline)\
README.md#build$(newline)\
docs/build.md$(newline)${newline})
