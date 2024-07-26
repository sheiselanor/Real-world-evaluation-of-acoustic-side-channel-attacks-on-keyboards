def sum_clicks_from_file(file_path):
    total_clicks = 0

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(':')
            if len(parts) == 2:
                try:
                    clicks = int(parts[1].strip())
                    total_clicks += clicks
                except ValueError:
                    print(f"Warning: Could not convert '{parts[1].strip()}' to an integer.")

    return total_clicks

# Correct path to your text file
text_file_path = r"E:\University\FYP\FYP_A\dataset\PREPROCESSING_DATA\phase1_038_s1\clickdata\4.txt"

# Calculate the total clicks
total_clicks = sum_clicks_from_file(text_file_path)
print(f"Total number of 38_1 clicks: {total_clicks}")
