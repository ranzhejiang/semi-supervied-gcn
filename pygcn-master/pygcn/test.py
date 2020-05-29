file_path = "../data/cora/cora.content"  # 文件路径
tmp = []
for line in open(file_path,'r'):
    tmp.append(line)

print(len(tmp))