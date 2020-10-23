# 信息检索第二次实验报告
## 201800130083 数据18 刁龙飞
--- 
### 实验目的：
- 实现最基本的Ranked retrieval model
- 使用SMART notation：lnc.ltc
- 改进Inverted index
- 选做支持所有的SMART notations
### 实验环境：
- 硬件环境：个人PC机
- 软件环境：Jupyter notebook Python3.6   
---
### 实验流程：
- 数据读取和数据预处理的方式与实验一相同，该部分内容可参照上一篇实验报告
#### 改进Inverted index：
- 实验一建立的Inverted index如下。为了便于排序，数据结构为嵌套列表。其中每个词项表示为一个列表，第0个元素为该词项名称，第1个元素使用列表记录出现该词项的文档频率和位置。
> [['emarket', [6, 30642168938889216, 30642236995674112, 30642254842433536, 30642412799922177, 30645241832800256, 30650642112450560]], ['embalm', [7, 309818178937180160, 309854719726211073, 310011565741068288, 312092255064322048, 312093702115979264, 312093978956808192, 312742661599686657]]]   
- 根据实验要求对实验一建立的倒排索引进行改进，改进后的数据结构如下。其中每个词项表示为一个列表，第0个元素为该词项名称和出现该词项的文档频率，第1个元素使用列表记录词项出现的文档和在该文档出现的次数。
> [[['email', 22], [[1, 29576869535817728], [1, 29898299192385538], [1, 303271814136729600], [1, 303646713581826049], [1, 307894083920207872], [1, 307894599823802369], [1, 307898622173908992], [1, 307902648680599552], [1, 307904263512809475], [1, 307912480133361664], [1, 307914334019936256], [1, 307936073126907907], [1, 307937528521060352], [1, 308261559489163265], [1, 308321307328667648], [1, 308333579899908096], [1, 308367981560356866], [1, 309080413454876672], [1, 310751122170212352], [1, 31771605050855424], [1, 623890778826235904], [1, 625014424483561472]]], [['eman', 2], [[1, 30067826060435456], [1, 30067834360963073]]], [['emanu', 3], [[1, 29615199631708161], [1, 297485981488128000], [1, 298460381217710080]]]]
#### 使用lnc.ltc实现最基本的Ranked retrieval model：
##### 对document进行处理：
- 建立词袋模型，使用向量表示文档。将每个文档表示成一个稀疏的向量，记录倒排索引中的每个词项在文档中出现的次数，即tf-raw。
- 使用python第三方库numpy可以实现对向量每个元素取对数，计算weight。
- numpy还提供了可供计算向量长度的方法，用于计算n'lized。
```python
def tf_log(array):
    tf_log_list = []
    for i in array:
        if i == 0:
            tf_log_list.append(0)
        else:
            tf_log_list.append(1+np.log10(i))
    return np.array(tf_log_list)
# SMART notation document
def doc_vector(term_list, inverted_index):#将文档根据倒排索引转化成一个稀疏的向量 有归一化
    term_dict = inverted_index_vector(inverted_index)
    term_score = []
    for term in term_list:#对于每个查询词项
        #计算term在term_list中的出现次数:tf-raw
        if term not in term_dict.keys():
            pass #对于所有文档中都没有出现过的词忽略不计
        else:
            term_dict[term] += 1
    for key in term_dict:
        term_score.append(term_dict[key])
        #归一化
    return tf_log(np.array(term_score)) / np.linalg.norm(tf_log(np.array(term_score)))
```
##### 对query进行处理：
- 建立类似的词袋模型，得到tf-raw。
- 取对数计算tf-wght。
- 从倒排索引中查询每个词项的df。
- 取对数计算idf。
- 最终的weight表示为tf*idf
```python
# SMART notation query
def term_list_vector(term_list, inverted_index):#对query处理 将term_list根据倒排索引转化成一个稀疏的向量 做tf和idf 不做归一化
    term_dict = inverted_index_vector(inverted_index)
    term_score = []
    df = []
    id_list = []
    for term in term_list:#对于每个查询词项
        #计算term在term_list中的出现次数:tf-raw
        if term not in term_dict.keys():
            pass #对于所有文档中都没有出现过的词忽略不计
        else:
            term_dict[term] += 1
    for i in inverted_index:#只检索与query有关的文档
        df.append(i[0][1])
        if i[0][0] in term_list:
            for j in i[1]:
                id_list.append(j[1])
    for key in term_dict:
        term_score.append(term_dict[key])
    tf_wght = tf_log(np.array(term_score))
    idf = tf_log(N/np.array(df))#计算tf,idf但是不进行归一化
    return tf_wght*idf,id_list#最后传出来这个参数是为了计算评分时方便，只计算id_list中的文档
```
##### 计算余弦相似度
- 计算文档集中每篇文档的得分，即sum(weight*n'lized)/len(document)
- 按照得分对文档进行排序，对排序后的文档取前K个打印到屏幕上。
```python
#时间复杂度略高
def cosine_score(query, document_list, inverted_index): #用预处理后的list
    scores = []#存放每篇文档的得分
    query_vector,id_list = term_list_vector(query, inverted_index)
    for i in document_list:
        if i[0] in id_list:
            document_vector = doc_vector(i[1:], inverted_index)
            scores.append([i[0],sum(query_vector*document_vector)/len(i)])    
    return sorted(scores,key = lambda x:x[1],reverse=True)
```
--- 
### 运行结果：
由于程序运行时间较长，取前2000条tweets，计算它们与query余弦相似度的得分。当query为hello world时，检索结果如下：
> 请输入查询语句：hello world   
共找到符合条件的结果26条，评分最高的前10条如下   
oprah's family secret weekly world news the weekly world news will not wait for the big o to divulge her litt http://bit.ly/id1qio   
world powers ponder options after failed iran nuclear talks world powers were considering the next step after t http://bit.ly/gep802   
sports digest fifa discusses moving 2022 world cup in qatar from summer to winter san jose http://bit.ly/e9y8ep fifa   
basic bedbug facts by joan wardthere are over nine hundred thousand species of insects all over the world which http://bit.ly/fcevev   
www.qatar vip.com qatar linking up for world cup qatar linking up for world cup http://bit.ly/fvfyhm   
just went to 2 mcdonalds that both said they were closed for 10 min apparently every mcd's in the world closes 10 min every day   
iran nuclear talks close with no progress the new york timestwo days of talks between iran and six world powers http://bit.ly/dhvtil   
dtn world news cantor i believe obama is a u.s citizen the new republican house majority leader says he does http://bit.ly/g6n09o   
piracy should not be tolerated by world powers it is very disturbing that there are 29 ships being held by soma http://bit.ly/eurns5   
nuclear talks between iran world powers fail http://su.pr/1z9zew fb
--- 
### 选做：支持所有的SMART notations
上面的做法仅适用于lnc.ltc这一种SMART notation，实际上还有很多。对上面的程序继续加以优化。原本对于query和ducument分开处理，分别使用lnc和ltc两种处理模式。我们可以整合成一个函数，将用户选择的操作模式作为参数。代码如下：
```python
#ALL SMART notations
def doc_vector(df, term_list, inverted_index, option):#将文档根据倒排索引转化成一个稀疏的向量 有归一化
    term_dict = inverted_index_vector(inverted_index)
    tf_raw = []
    id_list = []
    for term in term_list:#对于每个查询词项
        #计算term在term_list中的出现次数:tf-raw
        if term not in term_dict.keys():
            pass #对于所有文档中都没有出现过的词忽略不计
        else:
            term_dict[term] += 1
    for i in inverted_index:
        if i[0][0] in term_list:
            for j in i[1]:
                id_list.append(j[1])
    for key in term_dict:
        tf_raw.append(term_dict[key])
    option_list = [[np.array(tf_raw),tf_log(np.array(tf_raw)),(0.5 + (0.5*np.array(tf_raw)/max(tf_raw))),np.sign(tf_raw),(tf_log(np.array(tf_raw))/tf_log(np.full_like(np.array(tf_raw),np.mean(tf_raw),dtype = 'float')))],
                  [1,tf_log(N/np.array(df)),np.maximum(0,tf_log((N - np.array(df))/np.array(df)))],
                  [1,1 / np.linalg.norm(np.array(tf_raw))]]

    #option是一个字符串
    std_option = ['nlabL','ntp','nc']
    result = np.ones_like(tf_raw)
    for i in range(3):
        result =result * np.array(option_list[i][std_option[i].find(option[i])])
    return result,id_list
```
结果如下：
> 请输入查询语句：a big house   
请输入文档处理模式：anc   
请输入查询处理模式：bnn   
共找到符合条件的结果50条，评分最高的前10条如下   
house may kill arizona style immigration bill rep rick rand says the house is unlikely to pass the ari http://tinyurlcom/4jrjcdz   
oprah's family secret weekly world news the weekly world news will not wait for the big o to divulge her litt http://bit.ly/id1qio   
new easy recipe ranch house casserole http://bit.ly/htangc recipes   
producers guild gives the king's speech the big win http://st.ep2.tv/fbhsx2 film   
eric cantor addresses birther issue washington the new republican house majority leader says he doesn't thi http://twurl.nl/lr74ji   
dtn world news cantor i believe obama is a u.s citizen the new republican house majority leader says he does http://bit.ly/g6n09o   
understatement 8/5 runs in race 8 at big a for a 50g tag 6 year old horse with class but the tag sends mixed signals willing to lose him   
chinese pianist plays anti american tune at white house the blaze http://goo.gl/zu4jj   
@apoloohno if that horse races bet big but make sure the amount you bet is all 8's   
oprah winfrey says she's revealing a big family secret tomorrow on the oprah winfrey show http://bit.ly/hbtfpn   
--- 
附件.ipynb文件中的代码为支持所有SMART notations版本的代码。