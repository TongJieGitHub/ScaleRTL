import sys

if len(sys.argv) != 4:
    print("Usage: python3 convert_fatbin.py input.fatbin output.h module_name")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
module_name = sys.argv[3]

with open(input_file, "rb") as f:
    fatbin_data = f.read()

with open(output_file, "w") as f:
    f.write("#pragma once\n\n")
    f.write(f"const unsigned char kernelFatbin_{module_name}[] = {{\n")

    for i, byte in enumerate(fatbin_data):
        f.write(f"0x{byte:02x},")
        if (i + 1) % 16 == 0:
            f.write("\n")

    f.write("\n};\n")
    f.write(f"const size_t kernelFatbin_{module_name}_size = {len(fatbin_data)};\n")
