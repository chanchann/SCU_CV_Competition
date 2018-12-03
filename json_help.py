import json

'''
此函数作用为将
{"id":"15999.jpg","characters":"uFtN"}
转为为｛"49998.jpg": "gxMz"｝
方便label
存为train_data.json
'''

src_path='../train_annotation.json'
dest_path='../train_data.json'
def load(src_path):
    with open(src_path,'r') as f:
        data = json.load(f)
        return data

def store(data):
    with open(dest_path,'w') as f:
        f.write(json.dumps(data))

if __name__=='__main__':
    data=load(src_path)
    dict_map={}
    for i in range(len(data)):
        dict_map[data[i]['id']]=data[i]['characters']
    store(dict_map)

