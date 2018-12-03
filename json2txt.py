import json
import codecs
src_path='../train_annotation.json'
dest_path='../train.txt'
data = []
with open(src_path) as f:
    for line in f:
        data.append(json.loads(line))
file_object = codecs.open(dest_path, 'w' ,"utf-8")
for items in data:
    for item in items:
        str = "%s:%s\r\n" % (item['id'],item['characters'])
        file_object.write(str)
file_object.close()