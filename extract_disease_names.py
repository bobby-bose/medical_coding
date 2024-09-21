import csv
import json

def read_icd_codes(csv_file):
    icd_codes = {}
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            icd_code = row[2]
            disease_info = {
                "description": row[3],
                "full_description": row[4],
                "disease_name": row[5]
            }
            if icd_code in icd_codes:
                icd_codes[icd_code].append(disease_info)
            else:
                icd_codes[icd_code] = [disease_info]
    return icd_codes

# Example usage
csv_file_path = "ICD10codes.csv"
icd_codes = read_icd_codes(csv_file_path)

# Write to JSON file
output_file_path = "myapp/output.json"
with open(output_file_path, "w") as output_file:
    json.dump(icd_codes, output_file, indent=4)

print("Data has been written to output.json")
