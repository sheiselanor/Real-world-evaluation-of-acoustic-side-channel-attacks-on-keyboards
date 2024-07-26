# Read the file and process the lines
def process_clicks_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
        
    # Initialize variables
    unique_lines = []
    previous_count = None

    # Process lines
    for line in lines:
        parts = line.strip().split(':')
        if len(parts) == 2:
            click, count = parts
            if count != previous_count:
                unique_lines.append(line)
                previous_count = count

    # Write the unique counts to a new file
    with open(output_file, 'w') as file:
        for line in unique_lines:
            file.write(line)

# Replace 'input.txt' with the actual path to your input file
input_file_path = 'E:\\University\\FYP\\FYP_A\\dataset\\PREPROCESSING_DATA\\audio\\phase1_005_s1\\clickdata\\phase1_005_s1_clicknumber1.txt'
output_file_path = 'E:\\University\\FYP\\FYP_A\\dataset\\PREPROCESSING_DATA\\audio\\phase1_005_s1\\clickdata\\unique.txt'

process_clicks_file(input_file_path, output_file_path)
