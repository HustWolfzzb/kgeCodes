import time

from numpy import *
import operator

class Test:
    def __init__(self, entityList, entityVectorList,
                     typeList, typeVectorList,
                     relationList ,relationVectorList,
                     typeRelationList, typeRelationVectorList,
                     tripleListTrain, tripleListTest,
                 label = "head", isFit = False, k=5):
        self.entityList = {}
        self.relationList = {}
        for name, vec in zip(entityList, entityVectorList):
            self.entityList[name] = vec
        for name, vec in zip(relationList, relationVectorList):
            self.relationList[name] = vec
        self.tripleListTrain = tripleListTrain
        self.tripleListTest = tripleListTest
        self.rank = []
        self.label = label
        self.isFit = isFit
        self.ent2Type = {}
        self.typeList = {}
        if k>0:
            self.type_filter=True
        else:
            self.type_filter=False
        self.k = k
        self.typeRelationList = {}
        for name, vec in zip(typeList, typeVectorList):
            self.typeList[name] = vec
        for name, vec in zip(typeRelationList, typeRelationVectorList):
            self.typeRelationList[name] = vec
        self.get_ent2type()

    def writeRank(self, dir):
        print("写入")
        file = open(dir, 'w')
        for r in self.rank:
            file.write(str(r[0]) + "\t")
            file.write(str(r[1]) + "\t")
            file.write(str(r[2]) + "\t")
            file.write(str(r[3]) + "\n")
        file.close()


# ============================= ZZB =============================
    def get_ent2type(self):
        files = ['/Users/zhangzhaobo/PycharmProjects/ConnectE/data/FB15K/origin/FB15k_Entity_Type_train.txt',
                 '/Users/zhangzhaobo/PycharmProjects/ConnectE/data/FB15K/origin/FB15k_Entity_Type_test.txt',
                 '/Users/zhangzhaobo/PycharmProjects/ConnectE/data/FB15K/origin/FB15k_Entity_Type_valid.txt']
        for f in files:
            with open(f, 'r', encoding='utf8') as et:
                for line in et.readlines():
                    ent, type = line.strip().split()
                    if not self.ent2Type.get(ent):
                        self.ent2Type[ent] = [type]
                    else:
                        if type not in self.ent2Type[ent]:
                            self.ent2Type[ent].append(type)


    def calc_cos_sim(self, e1, e2):
        return e1.dot(e2) / (linalg.norm(e1) * linalg.norm(e2) )

    def calc_k_sim(self, types, r, label = 'tail'):
        k = self.k
        top_k = set()
        type_sim = {}
        for type in types:
            if label=='tail':
                h_r = array(self.typeList[type]) + array(self.typeRelationList[r])
            else:
                h_r = array(self.typeList[type]) - array(self.typeRelationList[r])
            alltype = list(self.typeList.keys())
            for t in alltype:
                 type_sim[t] = self.calc_cos_sim(h_r, array(self.typeList[t]))
            sorted_sim = sorted(type_sim.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            # if self.show:
            #     print(sorted_sim)
            #     self.show = False
            for x in range(k):
                top_k.add(sorted_sim[x][0])
            return top_k

    def type_suit(self, h, t, top_k, label = 'tail'):
        try:
            h_type = self.ent2Type[h]
            t_type = self.ent2Type[t]
            if label == 'head':
                for type in h_type:
                    if type in top_k:
                        return True
            else:
                for type in t_type:
                    if type in top_k:
                        return True
        except KeyError as e:
            return True
        return False

# ============================= ZZB =============================


    def getRank(self):
        start = time.process_time()
        cou = 0
        no_type = []
        all_hr_num=0
        all_tail_topk = readTypesConstriant('hr2types-complete.txt',sp=',',sp1=' ')
        print(len(all_tail_topk))
        # all_tail_topk = readTypesConstriant('hr2types-complete.txt',sp=',',sp1=' ')
        for triplet in self.tripleListTest:
            once = time.process_time()
            tail_topk = []
            # print("第 %s 个 Triple"%cou)
            if cou % 200 == 0:
                print("Time using: %s mins, Tested Triples:%s"%(round((time.process_time() - start)/60, 2), cou))
            rankList = {}
            pass_num=0
            try:
                if self.type_filter:
                    types = self.ent2Type[triplet[0]]
                    r = triplet[2]
                    for Type in types:
                        if not all_tail_topk.get((Type, triplet[2])):
                            print("居然还有错漏的？")
                            tail_topk = list(self.calc_k_sim([Type], triplet[2]))
                            all_tail_topk[(triplet[0], triplet[2])] = tail_topk
                            no_type.append((Type, triplet[2]))
                        else:
                            tail_topk = all_tail_topk.get((Type, triplet[2]))
                        all_hr_num+=1
            except KeyError as e:
                print(e)
                continue
            for entityTemp in self.entityList.keys():
                if self.label == "head":
                    corruptedTriplet = (entityTemp, triplet[1], triplet[2])
                    if self.isFit and (corruptedTriplet in self.tripleListTrain):
                        continue
                    rankList[entityTemp] = distance(self.entityList[entityTemp], self.entityList[triplet[1]], self.relationList[triplet[2]])
                else:#
                    if self.type_filter:
                        target = False
                        try:
                            for t in self.ent2Type[entityTemp]:
                                if t in tail_topk:
                                    target = True
                            # print("Tail: %s is skiping for Head: %s"%(entityTemp, triplet[0]))
                        except KeyError as e:
                            print(e)
                        if not target:
                            pass_num += 1
                            continue
                    corruptedTriplet = (triplet[0], entityTemp, triplet[2])
                    if self.isFit and (corruptedTriplet in self.tripleListTrain):
                        continue
                    rankList[entityTemp] = distance(self.entityList[triplet[0]], self.entityList[entityTemp], self.relationList[triplet[2]])
            cost = round(time.process_time()-once,2)
            if cost > 0.3:
                print(len(tail_topk),'Time:', round(time.process_time()-once,2),'pass_rate:',round(pass_num/len(self.entityList.keys()),3))
            # 根据第二个元素进行排序
            nameRank = sorted(rankList.items(), key = operator.itemgetter(1))
            if self.label == 'head':
                # 正确答案的索引位置
                numTri = 0
            else:
                numTri = 1
            x = 1
            for i in nameRank:
                if i[0] == triplet[numTri]:
                    break
                x += 1
            if x<=len(nameRank):
                self.rank.append((triplet, triplet[numTri], nameRank[0][0], x))
            # print(x)
            cou += 1
            # if cou % 10000 == 0:
            #     print("getRank" + str(cou))
        print("未知的类型头和关系长度%s, 所有h+r的数目%s"%(len(no_type), all_hr_num))
        print('Time Usage: ',time.process_time() - start)

    def outputTopK(self):
        start = time.process_time()
        cou = 0
        all_tail_type_topk = {}
        alltype = list(self.typeList.keys())
        for triplet in self.tripleListTest:
            # print("第 %s 个 Triple"%cou)
            if cou % 100 == 0:
                print("Time using: %s mins, Tested Triples:%s"%(round((time.process_time() - start)/60, 2), cou))
            cou+=1
            try:
                types = self.ent2Type[triplet[0]]
                top_k = []
                type_sim = {}
                r = triplet[2]
                for Type in types:
                    if all_tail_type_topk.get((Type, r)) != None:
                        continue
                    h_r = array(self.typeList[Type]) + array(self.typeRelationList[triplet[2]])
                    for t in alltype:
                        type_sim[t] = self.calc_cos_sim(h_r, array(self.typeList[t]))
                    sorted_sim = sorted(type_sim.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
                    # if self.show:
                    #     print(sorted_sim)
                    #     self.show = False
                    for x in range(20):
                        top_k.append(sorted_sim[x][0])
                    all_tail_type_topk[(Type, r)] = top_k
            except KeyError as e:
                print(e)
                continue
        with open('top5.txt','w',encoding='utf8') as top5:
            with open('top10.txt','w',encoding='utf8') as top10:
                with open('top15.txt','w',encoding='utf8') as top15:
                    with open('top20.txt','w',encoding='utf8') as top20:
                        files = [top5,top10,top15,top20]
                        num = [5,10,15,20]
                        for i in range(len(files)):
                            fi = files[i]
                            for key, value in all_tail_type_topk.items():
                                fi.write(key[0] +',' + key[1] + ','+' '.join(value[:num[i]])+'\n')

        print('Time Usage: ',time.process_time() - start)

    def getRelationRank(self):
        cou = 0
        self.rank = []
        for triplet in self.tripleListTest:
            rankList = {}
            for relationTemp in self.relationList.keys():
                corruptedTriplet = (triplet[0], triplet[1], relationTemp)
                if self.isFit and (corruptedTriplet in self.tripleListTrain):
                    continue
                rankList[relationTemp] = distance(self.entityList[triplet[0]], self.entityList[triplet[1]], self.relationList[relationTemp])
            nameRank = sorted(rankList.items(), key = operator.itemgetter(1))
            x = 1
            for i in nameRank:
                if i[0] == triplet[2]:
                    break
                x += 1
            self.rank.append((triplet, triplet[2], nameRank[0][0], x))
            # print(x)
            cou += 1
            if cou % 10000 == 0:
                print("getRelationRank" + str(cou))

    def getMeanRank(self):
        print("命中个数%s"%(len(self.rank)))
        num = 0
        for r in self.rank:
            num += r[3]
        return num/len(self.rank)


def distance(h, t, r):
    h = array(h)
    t = array(t)
    r = array(r)
    s = h + r - t
    # 二范数，平方和的根
    # 欧氏距离
    return linalg.norm(s)

def openD(dir, sp="\t"):
    #triple = (head, tail, relation)
    num = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            if num > 1000:
                break
            triple = line.strip().split(sp)
            if(len(triple)<3):
                continue
            list.append(tuple(triple))
            num += 1
    print("num", num)
    return num, list

def readTypesConstriant(file = 'hr2types-all.txt', sp=',',sp1='\t'):
    hr2types = {}
    with open(file) as f:
        lines = f.readlines()
        for l in lines:
            h,r,ts = l.strip().split(sp)
            hr2types[(h,r)] = ts.split(sp1)
    return hr2types

def combine_Topk_Constriant(file='top20.txt',sp=',',sp1=' '):
    topk = {}
    with open(file) as f:
        lines = f.readlines()
        for l in lines:
            h,r,ts = l.strip().split(sp)
            topk[(h,r)] = ts.split(sp1)
    hr2types = readTypesConstriant()
    print(len(hr2types), len(topk))
    overleap = 0
    hr_1 = {}
    with open('hr2types-complete.txt','w', encoding='utf8') as hr2txt:
        for hr in hr2types.keys():
            try:
                hr_1[hr] = list(set(hr2types[hr] + topk[hr]))
                overleap += 1
            except KeyError as e:
                hr_1[hr] = hr2types[hr]
            hr2txt.write(hr[0] + ',' + hr[1] + ',' + " ".join(hr_1[hr]) + '\n')
        for hr in topk.keys():
            if hr_1.get(hr) != None:
                continue
            try:
                hr_1[hr] = list(set(hr2types[hr] + topk[hr]))
                overleap += 1
            except KeyError as e:
                hr_1[hr] = topk[hr]
            hr2txt.write(hr[0] + ',' + hr[1] + ',' + " ".join(hr_1[hr]) + '\n')

    print("重叠个数:%s"%overleap)

def loadData(str):
    fr = open(str)
    sArr = [line.strip().split("\t") for line in fr.readlines()]
    datArr = [[float(s) for s in line[1][1:-1].split(", ")] for line in sArr]
    nameArr = [line[0] for line in sArr]
    return datArr, nameArr

if __name__ == '__main__':
    dirTrain = "data/FB15k/train.txt"
    # 获取训练三元组的数组
    tripleNumTrain, tripleListTrain = openD(dirTrain)
    dirTest = "data/FB15k/test.txt"
    # 获取测试三元组的数组
    tripleNumTest, tripleListTest = openD(dirTest)
    dirEntityVector = "data/FB15k/entity2vec.txt"
    # 加载TransE训练出来的实体向量
    entityVectorList, entityList = loadData(dirEntityVector)
    dirRelationVector = "data/FB15k/relation2vec.txt"
    # 加载TransE训练出来的关系向量
    relationVectorList, relationList = loadData(dirRelationVector)


    dirTypeVector = "data/FB15k/typeVector.txt"
    # 加载TransE训练出来的类型向量
    typeVectorList, typeList = loadData(dirTypeVector)

    typeRelationVector = "data/FB15k/typeRelationVector.txt"
    # 加载TransE训练出来的类型关系向量
    typeRelationVectorList, typeRelationList = loadData(typeRelationVector)

    print("开始 Test ...")

    # testHeadRaw = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest)
    # testHeadRaw.getRank()
    # print(testHeadRaw.getMeanRank())
    # testHeadRaw.writeRank("data/" + "testHeadRaw" + ".txt")
    # testHeadRaw.getRelationRank()
    # print(testHeadRaw.getMeanRank())
    # testHeadRaw.writeRank("data/" + "testRelationRaw" + ".txt")

    # testTailRaw = Test(entityList, entityVectorList, typeList, typeVectorList, relationList, relationVectorList, typeRelationList, typeRelationVectorList,  tripleListTrain, tripleListTest, label = "tail",k=20)
    # testTailRaw.outputTopK()
    # combine_Topk_Constriant()
    for k in [20]:
        testTailRaw = Test(entityList, entityVectorList, typeList, typeVectorList, relationList, relationVectorList, typeRelationList, typeRelationVectorList,  tripleListTrain, tripleListTest, label = "tail",k=k)
        # testTailRaw.outputTopK()
        testTailRaw.getRank()
        print(testTailRaw.getMeanRank())
        if k>0:
            testTailRaw.writeRank("data/" + "testTailRaw_filter_k" + str(k) + ".txt")
        else:
            testTailRaw.writeRank("data/" + "testTailRaw_origin" + ".txt")


    # testHeadFit = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, isFit = True)
    # testHeadFit.getRank()
    # print(testHeadFit.getMeanRank())
    # testHeadFit.writeRank("data/" + "testHeadFit" + ".txt")
    # testHeadFit.getRelationRank()
    # print(testHeadFit.getMeanRank())
    # testHeadFit.writeRank("data/" + "testRelationFit" + ".txt")
    #
    # testTailFit = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, isFit = True, label = "tail")
    # testTailFit.getRank()
    # print(testTailFit.getMeanRank())
    # testTailFit.writeRank("data/" + "testTailFit" + ".txt")