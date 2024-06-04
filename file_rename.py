import os
folder_path = "./img_data"
if __name__ == '__main__':
    field_sum = 0
    for filename in os.listdir(folder_path):
        if ("field" in filename):
            field_sum += 1
    for filename in os.listdir(folder_path):
        if ("Screenshot" in filename):
            new_filename = "field"+ '{:04d}'.format(field_sum) + ".png"
            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f"Renamed {filename} to {new_filename}")
            field_sum+=1