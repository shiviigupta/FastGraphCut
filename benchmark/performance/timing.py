import os
import subprocess

bb_path = "../dataset/large_bb.txt"
directory_path = "../dataset/large/"

bbox_dict = {}
with open(bb_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        filename = parts[0]
        bbox = list(map(int, parts[1:]))
        bbox_dict[filename] = bbox

km_fgd = 0
km_bgd = 0
calcbeta = 0
calcweights = 0

output_file = open("../dataset/timing_large.txt", "w")

file_list = os.listdir(directory_path)

for file in file_list:
    if file.endswith(".jpg"):
        file_path = os.path.join(directory_path, file)
        bbox = bbox_dict[file]

        print(f"\n\nProcessing {file_path} with bbox {bbox}...")

        #result = subprocess.run(["./SlowGrabCut", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = subprocess.run(["./SlowGrabCut", file_path, str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                 
        time = result.stdout.decode('utf-8')
        print(time)

        # Extract the relevant timing information from the output
        for i in time.split("\n"):
            if "background" in i:
                km_bgd += float(i.split()[-2])
            if "foreground" in i:
                km_fgd += float(i.split()[-2])
            if "Beta" in i:
                calcbeta += float(i.split()[-2])
            if "Weights" in i:
                calcweights += float(i.split()[-2])
        
        output_file.write(time)
        output_file.write("\n\n")

        # Here you can add the code to process the image
        # For example, you can call a function to perform graph cut on the image
        # graph_cut(file_path)
    

print(f"Total time for background: {km_bgd:.2f} us")
print(f"Total time for foreground: {km_fgd:.2f} us")
print(f"Total time for Beta calculation: {calcbeta:.2f} us")
print(f"Total time for Weights calculation: {calcweights:.2f} us")

output_file.write(f"Total time for background: {km_bgd:.2f} us\n")
output_file.write(f"Total time for foreground: {km_fgd:.2f} us\n")
output_file.write(f"Total time for Beta calculation: {calcbeta:.2f} us\n")
output_file.write(f"Total time for Weights calculation: {calcweights:.2f} us\n")
output_file.write("\n\n")
output_file.write(f"Total time for background per image (/50): {(km_bgd / 50):.2f} us\n")
output_file.write(f"Total time for foreground per image (/50): {(km_fgd / 50):.2f} us\n")
output_file.write(f"Total time for Beta calculation per image (/50): {(calcbeta / 50):.2f} us\n")
output_file.write(f"Total time for Weights calculation per image (/50): {(calcweights / 50):.2f} us\n")
output_file.close()