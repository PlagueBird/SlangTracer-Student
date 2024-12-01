import html

def convert_to_unicode(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # Replace the left and right apostrophes accordingly
        content = content.replace("'", "\\xe2\\x80\\x99").replace('"', "\\xe2\\x80\\x9c").replace("\n", "\\n")

        content = html.escape(content)

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(content)

    except Exception as e:
        print(f"An error occurred: {e}")

# Replacement Clause
input_html_file = 'sausage.html'
output_html_file = 'itmbtqq.html'
convert_to_unicode(input_html_file, output_html_file)