import os
import re
folder_path = "./img_data"
if __name__ == '__main__':
    pattern = re.compile(r'\d\d\d\d')
    max = 0
    for filename in os.listdir(folder_path):
        # Find all matches in the filename
        matches = pattern.findall(filename)
        
        for match in matches:
            print(matches)
            if ((match != "2024") and (int(match) > max)):
                max = int(match)
    max += 1
    print(max)
    for filename in os.listdir(folder_path):
        if ("Screenshot" in filename):
            new_filename = "field"+ '{:04d}'.format(max) + ".png"
            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f"Renamed {filename} to {new_filename}")
            max += 1