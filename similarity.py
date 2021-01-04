import numpy as np
import collections
# import argparse


def read_vectors_sim(vec_file):
    # input:  the file of word2vectors
    # output: word dictionay, embedding matrix -- np ndarray
    f = open(vec_file, 'r')
    embeddings = []
    for line in f:
        currLineListStr = line.strip().split("\t")
        currLineListFloat = []
        for i in currLineListStr:  # 逐行将字符串数据转化为浮点数
            currLineListFloat.append(float(i))
        embeddings.append(currLineListFloat)
    f.close()
    embeddings = np.array(embeddings)
    return embeddings



def read_type():
    ty_list = []
    f = open("/Users/zhangzhaobo/CLionProjects/DKRL/data/entity_word/type2id.txt", 'r')
    for line in f:
        data = line.split()
        ty_list.append(data[0])
    f.close()
    return ty_list

def sim_com(embeddings, k=10):
    # id_type = get_type2id()
    ty_list = read_type()
    # print(ty_list)

    cnt=len(embeddings)
    dim=len(embeddings[0])
    # fl=open('typeSim.txt', 'w')
    to_cnt = [0] * 10
    with open("top5.txt", 'w', encoding='utf8') as top5:
        with open("top10.txt", 'w', encoding='utf8') as top10:
            with open("top20.txt", 'w', encoding='utf8') as top20:
                files = [top5, top10, top20]
                ks = [5,10,20]
                for i in range(cnt):
                    sim=collections.OrderedDict()
                    for j in range(cnt):
                        if i == j:
                            continue
                        vsim = embeddings[i].dot(embeddings[j].T) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                        sim[j]=vsim
                    sorted_sim=sorted(sim.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
                    tmp=0
                    for s in range(len(files)):
                        files[s].write(str(i))
                        # files[s].write("top %d\t%s\t%f\n"%(tmp,ty_list[sorted_sim[k][0]],sorted_sim[k][1]))
                        for k in range(ks[s]):
                            files[s].write(' ' + str(sorted_sim[k][0]))
                        files[s].write('\n')
    print(cnt)
    for i in range(len(to_cnt)):
        print("%d\t%f"%(to_cnt[i],to_cnt[i]/cnt))

# vsim = embeddings[id1].dot(embeddings[id2].T) / (np.linalg.norm(embeddings[id1]) * np.linalg.norm(embeddings[id2]))

def get_type2id(filename='node-res/type2id.txt'):
    id_type = []
    with open(filename, 'r', encoding='utf8') as t2i:
        for line in t2i.readlines():
            type, id = line.strip().split()
            id_type.append(type)
    return id_type

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('file_name')
    # args = parser.parse_args()
    embeddings=read_vectors_sim('node-res/word2vec.bern')
    sim_com(embeddings, 10)