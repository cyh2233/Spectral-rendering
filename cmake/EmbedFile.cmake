# Embeds a binary/text file as a C byte array:
#   extern "C" const unsigned char <SYMBOL>[]; extern "C" const unsigned long long <SYMBOL>_size;
# Usage: cmake -DINPUT=<file> -DOUTPUT=<file.cpp> -DSYMBOL=<name> -P EmbedFile.cmake
list(GET INPUT 0 INPUT_FILE)
file(READ "${INPUT_FILE}" hex HEX)
string(LENGTH "${hex}" hex_len)
math(EXPR size "${hex_len} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],)(0x[0-9a-f][0-9a-f],)(0x[0-9a-f][0-9a-f],)(0x[0-9a-f][0-9a-f],)(0x[0-9a-f][0-9a-f],)(0x[0-9a-f][0-9a-f],)(0x[0-9a-f][0-9a-f],)(0x[0-9a-f][0-9a-f],))" "\\1\n" bytes "${bytes}")
file(WRITE "${OUTPUT}" "// Generated from ${INPUT_FILE}\n"
  "extern \"C\" const unsigned char ${SYMBOL}[] = {\n${bytes}0x00};\n"
  "extern \"C\" const unsigned long long ${SYMBOL}_size = ${size};\n")
