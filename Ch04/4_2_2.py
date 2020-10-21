improt math
# 计算每个标签的流行度

def TagPopularity(records):
    tagFreq = dict()
    for user, item, tag in records:
        if tag not in tagFreq:
            tagFreq[tag] = 1
        else:
            tagFreq[tag] += 1
    return tagFreq


# 物品i和j的余弦相似度  
def CosineSim(item_tags, i, j):
    ret = 0
    for b, wib  in item_tags[i].items():
        if b in item_tags[j]:
            ret += wib * item_tags[j][b]
    
    ni = 0
    hj = 0
    for b, w in item_tags[i].items():
        ni += w * w
    for b, w in item_tags[j].items():
        nj += w * w
    if ret == 0:
        return 0
    return ret/math.sqrt(ni*nj)


def Diversity(item_tags, recommend_items):
    ret = 0 
    n = 0
    for i in recommend_items.keys():
        for j in recommend_items.keys():
            if i == j:
                continue
            ret += CosineSim(item_tags, i, j)
            n += 1
    return ret /(n*1.0)

# 从records中统计出user_tags和tag_items
def InitStat(records):
    user_tags = dict()
    tag_items = dict()
    user_items = dict()
    for user, item, tag in records.items():
        addValueToMat(user_tags, user, tag, 1)
        addValueToMat(tag_items, tag, item, 1)
        addValueToMat(user_items, user, item, 1)

# 对用户进行个性化推荐  
def Recommend(user):
    recommend_items = dict()
    tagged_items = user_items[user]
    for tag, wut in user_tags[user].items():
        for item, wti in tag_items[tag].items():
            # 
            if  item in tagged_items:
                continue 
            if item not in recommend_items:
                recommend_items[item] = wut * wti 
            else:
                recommend_items[item] += wut * wti 
    return recommend_items

