import json

id2entity = []
id2rel = []

folder = "FB15k/"
bench_path = "data/" + folder
json_path = ""

with open(bench_path + "entity2id.txt", "r",encoding='utf8')as f:
    total = f.readlines()
    for content in total:
        e, i = content.strip().split("\t")
        id2entity.append(e)

with open(bench_path + "relation2id.txt", "r",encoding='utf8')as f:
    total = f.readlines()
    for content in total:
        e, i = content.strip().split("\t")
        id2rel.append(e)

fe = open(json_path + "typeVector-cos.txt", "w")
fr = open(json_path + "typeRelationVector-cos.txt", "w")

with open(json_path + "cos.json", 'r') as load_f:
    strF = load_f.read()
    if len(strF) > 0:
        load_dict = json.loads(strF)
        ent = load_dict['ent_embeddings']
        rel = load_dict['rel_embeddings']
        for i in range(len(ent)):
            fe.write("%s\t%s\n" % (id2entity[i], str(ent[i])))
        for i in range(len(rel)):
            fr.write("%s\t%s\n" % (id2rel[i], str(rel[i])))
fe.close()
fr.close()