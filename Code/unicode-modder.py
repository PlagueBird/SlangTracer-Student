import html
import sys

def convert_to_unicode(input_file, output_file):
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # Perform character replacements
        content = "b'" + content + "'"
        content = content.replace("‘","\\xe2\\x80\\x98")
        content = content.replace("’", "\\xe2\\x80\\x99")
        content = content.replace("\n", "\\n")
        content = content.replace("–", "\\xe2\\x80\\x93")
        content = content.replace("—", "\\xe2\\x80\\x94")

        # Write the modified content to the output file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"Conversion successful! Output written to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    convert_to_unicode(input_file, output_file)
