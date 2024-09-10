# Preprocess Script

This Python script filters lines from an input text file based on specified include and exclude filters. The filtering rules are defined in a `filters.json` file, which contains patterns in both plain text and regular expressions.

## Usage

To run the script, use the following command:

```
python preprocess.py <input_trace_text_file> <output_file> <filters.json>
```

- `<input_trace_text_file>`: Path to the input text file containing the trace data to be filtered.
- `<output_file>`: Path to the output file where the filtered results will be saved.
- `<filters.json>`: Path to the JSON file containing the include/exclude filters and default behavior.

## Filters Configuration

The filters are defined in a JSON file (`filters.json`) and consist of two main sections: `include` and `exclude`.

- **Include Filters**:  
  A list of text patterns or regular expressions used to identify lines that should be included in the output.

- **Exclude Filters**:  
  A list of text patterns or regular expressions used to identify lines that should be excluded from the output.

### Default Behavior

The `default` field in the `filters.json` file specifies how to handle lines that do not match any of the include or exclude filters. You can set it to either:

- `include`: All remaining lines (that do not match any exclude filters) will be included in the output.
- `exclude`: All remaining lines (that do not match any include filters) will be excluded from the output.
