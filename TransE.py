from random import uniform, sample
from numpy import *
from copy import deepcopy


class TransE:
    def __init__(self, entityList, relationList, tripleList, margin=1, learingRate=0.00001, dim=10, L1=True):
        self.margin = margin
        self.learingRate = learingRate
        self.dim = dim  # 向量维度
        self.entityList = entityList  # 一开始，entityList是entity的list；初始化后，变为字典，key是entity，values是其向量（使用narray）。
        self.relationList = relationList  # 理由同上
        self.tripleList = tripleList  # 理由同上
        self.loss = 0
        self.L1 = L1

    # def show(self):
    #     Sbatch = self.getSample(150)
    #     print(Sbatch)

    def initialize(self):
        '''
        初始化向量
        '''
        entityVectorList = {}
        relationVectorList = {}
        for entity in self.entityList:
            n = 0
            entityVector = []
            while n < self.dim:
                # uniform(-6 / (dim ** 0.5), 6 / (dim ** 0.5))
                ram = init(self.dim)  # 初始化的范围
                entityVector.append(ram)
                n += 1
            entityVector = norm(entityVector)  # 归一化
            entityVectorList[entity] = entityVector
        print("entityVector初始化完成，数量是%d" % len(entityVectorList))
        for relation in self.relationList:
            n = 0
            relationVector = []
            while n < self.dim:
                # uniform(-6 / (dim ** 0.5), 6 / (dim ** 0.5))
                ram = init(self.dim)  # 初始化的范围
                relationVector.append(ram)
                n += 1
            relationVector = norm(relationVector)  # 归一化
            relationVectorList[relation] = relationVector
        print("relationVectorList初始化完成，数量是%d" % len(relationVectorList))
        self.entityList = entityVectorList
        self.relationList = relationVectorList

    def transE(self, cI=20):
        print("训练开始")
        for cycleIndex in range(cI):
            # 获取样本sample三元组 [(),()]
            Sbatch = self.getSample(150)
            Tbatch = []  # 元组对（原三元组，打碎的三元组）的列表 ：{((h,r,t),(h',r,t'))}
            for sbatch in Sbatch:
                # 获取一个被随机置换了头 or 尾 实体的三元组作为负例
                tripletWithCorruptedTriplet = (sbatch, self.getCorruptedTriplet(sbatch))
                if (tripletWithCorruptedTriplet not in Tbatch):
                    Tbatch.append(tripletWithCorruptedTriplet)
            self.update(Tbatch)
            if cycleIndex % 100 == 0:
                print("第%d次循环" % cycleIndex)
                print(self.loss)
                self.writeRelationVector("data/FB15k/relationVector.txt")
                self.writeEntilyVector("data/FB15k/entityVector.txt")
                self.loss = 0

    def getSample(self, size):
        return sample(self.tripleList, size)

    def getCorruptedTriplet(self, triplet):
        '''
        training triplets with either the head or tail replaced by a random entity (but not both at the same time)
        :param triplet:
        :return corruptedTriplet:
        '''
        i = uniform(-1, 1)
        if i < 0:  # 小于0，打坏三元组的第一项
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]
                if entityTemp != triplet[0]:
                    break
            corruptedTriplet = (entityTemp, triplet[1], triplet[2])
        else:  # 大于等于0，打坏三元组的第二项
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]
                if entityTemp != triplet[1]:
                    break
            corruptedTriplet = (triplet[0], entityTemp, triplet[2])
        return corruptedTriplet

    def update(self, Tbatch):
        copyEntityList = deepcopy(self.entityList)
        copyRelationList = deepcopy(self.relationList)

        for tripletWithCorruptedTriplet in Tbatch:
            # tripletWithCorruptedTriplet是原三元组和打碎的三元组的元组tuple
            # 正例头结点
            headEntityVector = copyEntityList[
                tripletWithCorruptedTriplet[0][0]]
            # 正例尾结点
            tailEntityVector = copyEntityList[tripletWithCorruptedTriplet[0][1]]
            # 关系
            relationVector = copyRelationList[tripletWithCorruptedTriplet[0][2]]
            # 负例头结点
            headEntityVectorWithCorruptedTriplet = copyEntityList[tripletWithCorruptedTriplet[1][0]]
            # 负例尾结点
            tailEntityVectorWithCorruptedTriplet = copyEntityList[tripletWithCorruptedTriplet[1][1]]

            # 初始的三元组的头结点
            headEntityVectorBeforeBatch = self.entityList[
                tripletWithCorruptedTriplet[0][0]]  # tripletWithCorruptedTriplet是原三元组和打碎的三元组的元组tuple
            # 初始的三元组的尾结点
            tailEntityVectorBeforeBatch = self.entityList[tripletWithCorruptedTriplet[0][1]]
            # 初始的三元组的关系
            relationVectorBeforeBatch = self.relationList[tripletWithCorruptedTriplet[0][2]]
            # 初始的负例三元组的头结点
            headEntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][0]]
            # 初始的负例三元组的尾结点
            tailEntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][1]]

            # default: True
            if self.L1:
                # fabs(h + r - t).sum()
                # 取初始的头尾节点和关系做 h + r - t
                distTriplet = distanceL1(headEntityVectorBeforeBatch, tailEntityVectorBeforeBatch,
                                         relationVectorBeforeBatch)
                # 取 负例的头尾节点和关系做 h' + r' - t'
                distCorruptedTriplet = distanceL1(headEntityVectorWithCorruptedTripletBeforeBatch,
                                                  tailEntityVectorWithCorruptedTripletBeforeBatch,
                                                  relationVectorBeforeBatch)
            else:
                # ( ( h+r-t) * (h+r-t) ).sum()
                distTriplet = distanceL2(headEntityVectorBeforeBatch, tailEntityVectorBeforeBatch,
                                         relationVectorBeforeBatch)
                distCorruptedTriplet = distanceL2(headEntityVectorWithCorruptedTripletBeforeBatch,
                                                  tailEntityVectorWithCorruptedTripletBeforeBatch,
                                                  relationVectorBeforeBatch)
            # 计算出相应的差值 叠加 定制margin
            # where [x] +denotes the positive part of x, γ > 0 is a margin hyperparameter,
            # 在算法的倒数第二行，有一个[x]+的用法
            eg = self.margin + distTriplet - distCorruptedTriplet
            # 如果正例 和 负例的距离之差 大于 - margin 那么就叠加到loss里面去  梯度计算：https://blog.csdn.net/MonkeyDSummer/article/details/85253813
            if eg > 0:
                # 如果差距过小,导致整体的损失过小, 说明正例和负例隔得不够开, 直接舍弃这一次迭代
                self.loss += eg
                # 二范式距离的梯度为 2(h + r -t)不用考虑正负, 一范式则要考虑正负以及转化为求导后的+-1
                tempPositive = 2 * self.learingRate * (
                        tailEntityVectorBeforeBatch - headEntityVectorBeforeBatch - relationVectorBeforeBatch)
                # 初始负例的 t - (h + r), 作为负例的反向梯度
                tempNegtative = 2 * self.learingRate * (
                        tailEntityVectorWithCorruptedTripletBeforeBatch - headEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch)

                if self.L1:
                    # 如果采用一范数作为distance，那么就得考虑正负问题，因为一范数是 | h + r - t| 如果为负，则需要转换为 t - r - h, 求导后为+-1
                    tempPositiveL1 = []
                    tempNegtativeL1 = []
                    for i in range(self.dim):  # 不知道有没有pythonic的写法（比如列表推倒或者numpy的函数）？
                        if tempPositive[i] >= 0:
                            tempPositiveL1.append(1)
                        else:
                            tempPositiveL1.append(-1)

                        if tempNegtative[i] >= 0:
                            tempNegtativeL1.append(1)
                        else:
                            tempNegtativeL1.append(-1)
                    tempPositive = array(tempPositiveL1)
                    tempNegtative = array(tempNegtativeL1)
                # * 头结点 + 正梯度
                # * 尾结点 - 正梯度
                # * 关系 + 正梯度  - 负梯度
                headEntityVector = headEntityVector + tempPositive
                tailEntityVector = tailEntityVector - tempPositive
                relationVector = relationVector + tempPositive - tempNegtative
                # * 负例头结点 - 负梯度
                # * 负例尾结点 + 负梯度
                headEntityVectorWithCorruptedTriplet = headEntityVectorWithCorruptedTriplet - tempNegtative
                tailEntityVectorWithCorruptedTriplet = tailEntityVectorWithCorruptedTriplet + tempNegtative

                # 只归一化这几个刚更新的向量，而不是按原论文那些一口气全更新了
                copyEntityList[tripletWithCorruptedTriplet[0][0]] = norm(headEntityVector)
                copyEntityList[tripletWithCorruptedTriplet[0][1]] = norm(tailEntityVector)
                copyRelationList[tripletWithCorruptedTriplet[0][2]] = norm(relationVector)
                copyEntityList[tripletWithCorruptedTriplet[1][0]] = norm(headEntityVectorWithCorruptedTriplet)
                copyEntityList[tripletWithCorruptedTriplet[1][1]] = norm(tailEntityVectorWithCorruptedTriplet)

        self.entityList = copyEntityList
        self.relationList = copyRelationList

    def writeEntilyVector(self, dir):
        # print("写入实体")
        entityVectorFile = open(dir, 'w')
        for entity in self.entityList.keys():
            entityVectorFile.write(entity + "\t")
            entityVectorFile.write(str(self.entityList[entity].tolist()))
            entityVectorFile.write("\n")
        entityVectorFile.close()

    def writeRelationVector(self, dir):
        # print("写入关系")
        relationVectorFile = open(dir, 'w')
        for relation in self.relationList.keys():
            relationVectorFile.write(relation + "\t")
            relationVectorFile.write(str(self.relationList[relation].tolist()))
            relationVectorFile.write("\n")
        relationVectorFile.close()


def init(dim):
    return uniform(-6 / (dim ** 0.5), 6 / (dim ** 0.5))


def distanceL1(h, t, r):
    s = h + r - t
    sum = fabs(s).sum()
    return sum


def distanceL2(h, t, r):
    s = h + r - t
    sum = (s * s).sum()
    return sum


def norm(list):
    '''
    归一化
    :param 向量
    :return: 向量的平方和的开方后的向量
    '''
    var = linalg.norm(list)
    i = 0
    while i < len(list):
        list[i] = list[i] / var
        i += 1
    return array(list)


def openDetailsAndId(dir, sp="\t"):
    idNum = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            list.append(DetailsAndId[0])
            idNum += 1
    return idNum, list


def openTrain(dir, sp="\t"):
    num = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if (len(triple) < 3):
                continue
            list.append(tuple(triple))
            num += 1
    return num, list


if __name__ == '__main__':
    dirEntity = "data/FB15k/entity2id.txt"
    # 读取实体id 以及id所对应的实体列表
    entityIdNum, entityList = openDetailsAndId(dirEntity)
    dirRelation = "data/FB15k/relation2id.txt"
    # 读取关系id 以及id所对应的关系列表
    relationIdNum, relationList = openDetailsAndId(dirRelation)
    dirTrain = "data/FB15k/train.txt"
    tripleNum, tripleList = openTrain(dirTrain)
    print("打开TransE...")
    transE = TransE(entityList, relationList, tripleList, margin=1, dim=100)
    print("TranE初始化...")
    transE.initialize()
    transE.transE(1000)
    transE.writeRelationVector("data/FB15k/relationVector.txt")
    transE.writeEntilyVector("data/FB15k/entityVector.txt")
