import os

def merge_files(output_file, input_prefix):
    # 获取所有分片文件并按顺序排序 (aa, ab, ac...)
    parts = sorted([f for f in os.listdir('.') if f.startswith(input_prefix)])
    
    with open(output_file, 'wb') as outfile:
        for part in parts:
            print(f"Merging {part}...")
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
    print(f"Done! Created {output_file}")

if __name__ == "__main__":
    # 将 model.safetensors.part_aa 等合并回 model.safetensors
    merge_files("model.safetensors", "model.safetensors.part_")