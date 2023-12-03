import os

# 读取原始文件
file = "word_test.txt"
path =os.path.join("word",file)

with open(path, 'r') as f:
    lines = f.readlines()

# 定义字段映射关系
field_mapping = {
    "耳鼻咽喉科": "其他",
    "影像检验科": "其他",
    "血液科": "其他",
    "口腔科": "其他",
    "中医科": "其他",
    "烧伤科": "其他",
    "风湿免疫科": "其他",
    "精神心理科": "其他",
    "整形科": "其他",
}
categories = set()  # 用于记录不同的种类
# 修改第一个字段并生成新文件内容
new_lines = []
for line in lines:
    # 按照一定规则修改第一个字段
    fields = line.split('\t')  # 假设字段之间使用制表符分隔
    first_field = fields[0]

    # 判断第一个字段是否包含"内科"
    if "内科" in first_field:
        modified_field = "内科"
    elif "外科" in first_field:
        modified_field = "外科"
    elif first_field in field_mapping:
        modified_field = field_mapping[first_field]
    else:
        modified_field = first_field

    fields[0] = modified_field

    new_line = '\t'.join(fields)  # 将修改后的字段重新组合成一行
    new_lines.append(new_line)

    categories.add(modified_field)  # 将修改后的字段添加到集合中

print(len(categories))
print(categories)
# 将新文件内容写入新文件
with open(file, 'w') as f:
    f.writelines(new_lines)
