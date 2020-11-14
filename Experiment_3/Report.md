# 信息检索第三次实验报告
## 201800130083 数据18 刁龙飞
--- 
### 实验目的：
- 理解老师给出的代码
- 补全代码，完成MAP、MRR和NDCG三种评价指标
- 在提供的仿真数据集上完成三种指标的评价
### 实验环境：
- 硬件环境：个人PC机
- 软件环境：Jupyter notebook, Spyder Python3.6   

### 说明：在11.13实验课验收之后对代码进行了调整。报告中调整前的错误内容保留但以~~删除线~~显示。报告中展示的代码均为最终的正确代码
---
### 实验流程：
#### 对于已给出的代码的理解：
最原始的数据集包括tweets.txt和qrels2014.txt，表示所有的tweets和基于tweets的查询结果。在qrels2014.txt文件中每一行按照空格分割有四列，分别表示查询编号、类型、相关tweet编号、关联值。关联值有0，1，2三种，会在计算NDCG评价指标时用到。   
>171 Q0 307390255743528960 2   
171 Q0 307431003394281472 2   
171 Q0 297987456635117569 0   
171 Q0 307496547812270080 2   
171 Q0 307364469183483906 2   
171 Q0 307381804225163264 1   
171 Q0 298537912893595649 0   
171 Q0 307600151294382080 2   

qrels.txt是对quels2014.txt处理而来，剔除了quels2014.txt中相关tweet编号不在tweets.txt中的行。我们将该数据集用作事实上的相关性标准，对结果进行评价。

result.txt为一个仿真的结果，意在模拟某个信息检索系统对于相关quel得到的相关性文档的结果。我们的目的是采用MAP、MRR和NDCG三种评价指标对该结果进行评价。

已给出的代码包括两个文件，第一个文件process.py用于对原始数据集tweets.txt和qrels2014.txt进行预处理及模拟仿真，并得到qrels.txt和result.txt两个结果。evalution.py用于补全代码，对result.txt的结果加以评价，输出评价值。

下面补充的函数参数说明：
>qrels_dict = {query_id:{doc_id:gain, doc_id:gain, ...}, ...}   
>test_dict = {query_id:[doc_id, doc_id, ...], ...}   
>k为阈值，即取返回到的结果的前k个进行评价。用户往往不关注靠后的检索结果。默认k=100

#### 使用MAP指标：
对于每个查询，遍历r从1到result对应的文档长度。~~将前r个结果中的正确率求平均，得到Average Precision。最后将得到的所有的Average Precision再求平均，得到MAP。~~  
若第r个文档是相关的，则计算其正确率。将前r个结果中的正确率求平均，得到Average Precision。要注意求平均时分母是相关文档的个数。

```python
def MAP_eval(qrels_dict, test_dict, k = 100):
    #每个quel的precision的平均
    map_list = []
    for quel in qrels_dict:
        k_list = []
        l = min(len(qrels_dict[quel].keys()),len(test_dict[quel]),k)
        quel_docs = [i for i in set(qrels_dict[quel].keys())]
        for r in range(l): #前r个文档遍历
            if test_dict[quel][r] in quel_docs:
                k_list.append(len([i for i in test_dict[quel][:r+1] if i in quel_docs])/(r+1))#相关文档统计正确率
        map_list.append(np.sum(np.array(k_list))/len([i for i in test_dict[quel][:l] if i in quel_docs]))
    return np.mean(np.array(map_list))
```
在这几个函数计算过程中还需要注意"前k个文档"的取值。理论上对前k个计算，但在实现过程中要考虑结果集和事实集大小的影响。因此对于每个查询，要在这三者中取最小。

#### 使用MRR指标：
对于每个查询，取相应查询集中的前k个文档，关注这些文档在结果集中出现的位置，取最高的那一个，对该位置取倒数作为本次查询的得分。最后对所有查询的得分取平均。
~~对于每个查询，只关注与它相关的第一个文档，找到这个文档在查询结果中的位置，对该位置取倒数。~~  
需要注意的是，由于python的位置编号从0开始，这里需要+1。
```python
def MRR_eval(qrels_dict, test_dict, k = 100):
    K_list = []
    for quel in qrels_dict:
        First_doc = list(qrels_dict[quel].keys())[:k]
        K = 100
        for doc in First_doc:
            K = min(K,test_dict[quel].index(doc))
        K_list.append(K)
    return np.mean(1/(np.array(K_list)+1))
```

#### 使用NDCG指标：
对于每个查询，首先对于返回的结果文档集前k个计算DCG。我采用的数据结构是numpy提供的array类型。由于返回的结果文档集可能是不相关的，此时的相关性就是0。每次查询得到一个DCG，为了将其归一化，我们需要求出理想状况下的DCG即IDCG。   

由于gain的取值只有0,1,2，而相关的只有1,2。~~我们将与该查询相关的文档按照gain进行排序~~  我们将事实集上的前k个文档进行排序。得到在给定相关的文档集上计算IDCG，使用IDCG对DCG进行归一化得到NDCG。

最后所有查询的NDCG求平均。  
计算DCG和IDCG有多种不同的方法，下面的代码使用较为通用的一种计算方式：  
参考资料：https://www.cnblogs.com/by-dream/p/9403984.html
```python
def NDCG_eval(qrels_dict, test_dict, k = 100):
    NDCG = []
    for quel in qrels_dict:
        rel = []
        i = np.arange(1,k+1)
        for j in test_dict[quel][:k]:
            try:
                rel.append(qrels_dict[quel][j])
            except KeyError:
                rel.append(0)
        #可能不到k个结果
        l = min(len(test_dict[quel][:]),k)
        i = np.arange(1,l+1)
        DCG_list = np.array(rel)/np.log2(i+1)#不能是0
        DCG = np.sum(DCG_list)
        #排序
        quel_docs = [i for i in list(qrels_dict[quel].keys())][:l]
        sorted_list = [i for i in quel_docs if qrels_dict[quel][i] == 2] +\
        [i for i in quel_docs if qrels_dict[quel][i] == 1]
        
        IDCG_rel = []
        for j in sorted_list:
            IDCG_rel.append(qrels_dict[quel][j])
        I = np.arange(1,len(sorted_list)+1)
        IDCG_list = np.array(IDCG_rel)/np.log2(I+1)#不能是0
        IDCG = np.sum(IDCG_list)
        NDCG.append(DCG/IDCG)
    return np.mean(np.array(NDCG))
```

严格按照PPT的公式计算，代码如下：
```python
def NDCG_eval(qrels_dict, test_dict, k = 100):
    NDCG_k = []
    for quel in qrels_dict:
        l = min(len(qrels_dict[quel].keys()),len(test_dict[quel]),k)
        rel = []
        i = np.arange(1,k+1)
        for j in test_dict[quel][:l]:
            try:
                rel.append(qrels_dict[quel][j])
            except KeyError:
                rel.append(0)
        #可能不到k个结果
        DCG_k = rel[0] + sum(np.array(rel[1:])/np.log2(np.arange(2,l+1)))
        #排序
        quel_docs = [i for i in list(qrels_dict[quel].keys())][:l]
        sorted_list = [i for i in quel_docs if qrels_dict[quel][i] == 2] + [i for i in quel_docs if qrels_dict[quel][i] == 1]
        
        IDCG_rel = []
        for j in sorted_list:
            IDCG_rel.append(qrels_dict[quel][j])
        IDCG_k = IDCG_rel[0] + sum(np.array(IDCG_rel[1:])/np.log2(np.arange(2,l+1)))#不能是0
        NDCG_k.append(DCG_k/IDCG_k)
    return np.mean(np.array(NDCG_k))
```

结果如下：
>MAP = 0.8701836509684747   
MRR = 0.79737012987013   
NDCG = 0.8666625607201335   

NDCG按照PPT的公式计算：
>NDCG = 0.8108022167379387
---
### 实验心得：
通过学习老师的代码有助于改善自己的编程风格。通过这次实验复习了信息检索的相关评价指标，也练习了相关数据结构的使用。   
在实验的验收过程中，发现自己对于这三种评价指标的理解均有一定的偏差。课后对代码进行了修正。