import html
import sys
import re

def convert_to_unicode(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # Remove emojis using the regex from Code A
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
        content = emoji_pattern.sub(r'', content)  # Remove

        content = "b'" + content + "'"
        content = content.replace("‘", "\\xe2\\x80\\x98")
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

def byte_bite(input_file, output_file):
    try:
        # Read the input file as bytes
        with open(input_file, 'rb') as file:
            content = file.read()

        # Remove occurrences of 0x9d (if you want to remove this specific byte)
        content = content.replace(bytes([0x9d]), b'')  # Remove 0x9d byte

        # Decode back to UTF-8
        content = content.decode('utf-8', errors='ignore')  # Ignore any other errors

        # Write the cleaned content to the output file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"File processed successfully! Output written to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    convert_to_unicode(input_file, output_file)
    byte_bite(output_file, output_file)