import requests

def fetch_html_dump(url, output_file):

    try:
        print(f"Fetching HTML from: {url}")
        response = requests.get(url)

        if response.status_code == 200:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(response.text)
            print(f"HTML successfully saved to {output_file}")
        else:
            print(f"Failed to fetch page. HTTP status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Replace with the desired URL
    target_url = input("Enter the URL to fetch: ").strip()

    # Replace with the desired output file name
    output_file_name = input("Enter the output file name (e.g., output.html): ").strip()

    fetch_html_dump(target_url, output_file_name)