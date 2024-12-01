def get_character_at_position(file_path, position):
    """
    Reads the content of the given file and returns the character at the specified position.

    Parameters:
        file_path (str): Path to the HTML file.
        position (int): The position (0-indexed) of the character to retrieve.

    Returns:
        str: The character at the specified position, or an error message if out of range.
    """
    try:
        # Read the HTML file content
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Check if the position is valid
        if 0 <= position < len(content):
            return content[position]
        else:
            return f"Position {position} is out of range. The file contains {len(content)} characters."
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    # Input file path
    file_path = input("Enter the path to the HTML file: ").strip()

    # Input position
    try:
        position = int(input("Enter the position number (0-indexed): ").strip())
        result = get_character_at_position(file_path, position)
        print(f"Result: {result}")
    except ValueError:
        print("Invalid position number. Please enter a valid integer.")
