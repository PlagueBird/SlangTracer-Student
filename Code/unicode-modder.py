import html

def convert_to_unicode(input_file, output_file):
    try:
        #Read
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 
        content_blob = ""
        for char in content:
            char.replace("'", "\\xe2\\x80\\x99").replace("'", "\\xe2\\x80\\x98").replace("\n", "\\n")

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(content_blob)

    except Exception as e:
        print(f"An error occurred: {e}")

input_html_file = '3e7vqxq.html'  #Replace
output_html_file = '3e7vqxq.html' #Replace
convert_to_unicode(input_html_file, output_html_file)