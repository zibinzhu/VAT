import random
import os

# 定义文件路径
val_file_path = "/data/zhuzibin/Workspace/Python/Dataset/THuman/Thuman2.1_1024/divided/val.txt"

# 读取现有的 val.txt 文件中的数据
with open(val_file_path, 'r') as f:
    existing_data = f.read().splitlines()

# 将现有的数据转换为集合以去重
existing_set = set(existing_data)

# 定义数据范围
total_data = {f"{i:04d}" for i in range(2446)}  # 编号从 0000 到 2445
remaining_data = sorted(total_data - existing_set)  # 剩余数据

# 随机选择 147 条数据
num_to_select = 147
selected_data = random.sample(remaining_data[525:], num_to_select)  # 从 525 之后的数据中选择

# 将新的选择数据写入 val.txt
with open(val_file_path, 'a') as f:
    for item in selected_data:
        f.write(item + '\n')

print(f"Selected {num_to_select} new data and appended to {val_file_path}.")