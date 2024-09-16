import re
import json
import argparse
import subprocess
import os

# Load filters from external JSON file
def load_filters(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        return json.load(file)

# Check if a line matches any regex in a list of patterns
def matches_any_pattern(line, patterns):
    return any(pattern.search(line) for pattern in patterns)

# Process the large file line by line using regex filters
def process_large_file_line_by_line(input_file, output_file, filters_file):
    # Load include/exclude filters

    filters = load_filters(filters_file)

    include_patterns = [re.compile(pattern) for pattern in filters["include"]]
    exclude_patterns = [re.compile(pattern) for pattern in filters["exclude"]]
    default_action = filters.get("default", "include")

    with open(input_file, 'r', encoding='utf-16') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Skip lines that match any of the exclude patterns
            if matches_any_pattern(line, exclude_patterns):
                continue
            # Write lines that match any of the include patterns
            if matches_any_pattern(line, include_patterns):
                outfile.write(line)
            elif default_action == "include":
                # Write lines that do not match any include or exclude filters if default is include
                outfile.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a large file with regex filters.')
    parser.add_argument('input_file', type=str, help='Path to the large input file.')
    parser.add_argument('output_file', type=str, help='Path to the filtered output file.')
    parser.add_argument('filters_file', type=str, help='Path to the JSON file with include/exclude filters.')

    args = parser.parse_args()
    command = f'pktmon etl2txt "{args.input_file}"'
    # Execute the PowerShell command
    subprocess.run(["powershell", "-Command", command], capture_output=True, text=True)
    txt_file = os.path.splitext(args.input_file)[0] + ".txt"
    # Process the large file using line-by-line approach

    process_large_file_line_by_line(txt_file, args.output_file, args.filters_file)

    print("Processing complete. Check the filtered output file.")
