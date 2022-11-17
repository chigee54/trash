import json
import numpy as np
import torch
from numpy import mean

'''
"ridx": 1355, 
"q": "2003年至2007年7月，被告人侯金凯在原成都铁路分局任行管员和成都铁路办事处任调查员期间，主要负责管理、出租单位在原成都铁路分局前门、后门的商铺，
官某某（另案处理）协助侯金凯收取租金。2007年7月被告人侯金凯退休后，仍回原单位协助官某某负责商铺的管理工作。2008年9月成都铁路办事处撤销，
被告人侯金凯仍以成都铁路办事处工作人员的身份收取原成都铁路分局前门、后门四间商铺（成都荷花池七区A40、41、42三间以及分局大门左边第六间商铺）的租金，
并向承租户出具盖有已作废公章的收据，先后收取本应属成都铁路局所有的租金132300元。案发后，被告人家属向侦查机关退赃100000元。", 
"crime": ["诈骗罪"]}

ajName":"王铮贪污、挪用公款案",
"ajjbqk":"大连市中级人民法院经公开审理查明： 1997年12月16日，大连市人民体育场（以下简称体育场）为大连市体育局下属的事业单位，
被告人王铮利用担任体育场场长，主管财务的职务便利，擅自以给该场副场长朱可冬购房为由，通过单位会计徐作德套取一张45万元的转账支票，并在该支票根上冒用朱的签名。
嗣后将该支票存人大连康泰建筑技术咨询有限公司（以下简称康泰公司）预收款账户。随后，王铮委托他人将该45万元从康泰公司转出并提取现金。
1999年5月，被告人王铮将该45万元现金用于注册以其妻子和女儿为股东的个人企业大连广鸿经贸有限公司的验资款，之后被告人又将该45万元作为广鸿公司租赁体育场看台的租金支付给体育场。 
2000年上半年，被告人王铮向大连市体育运动委员 （以下简称市体委）主任辛德智及辽宁省体育彩票管理中心（以下简称省体彩中心）主任邢立泉谎称其没有分到福利住房，要求解决住房。
2000年7月3日，市体委产业处处长包伟堂根据辛德智的旨意，以“一直没有兑现给王铮奖励一套住房”为由，经市体委给省体彩中心打了“关于奖励王铮、王国胜同志住房的请示”（以下简称请示）报告。
同年7月8日，省体彩中心主任邢立泉在该“请示”上批示：“同意用应兑现奖金为二位同志解决住房。”王铮将该“请示”出示给辛德智看后自行保存。
同年8—11月期间，王铮找到大连凤元装饰有限公司（以下简称凤元公司）的丁学春，让丁以支付大连市体育彩票管理中心（以下简称市体彩中心）装修款的名义，
分别开出30万元、18万元的装修款假发票各一张，交给辛德智签名。后王铮以辛批准的省体彩中心奖励购房款的名义，将该发票交给大连体育场改造工程指挥部（以下简称指挥部）出纳员李光怀，
从李处领取30万元、18万元转账支票各一张。同年9月20日、11月14日，王铮将这两张转账支票存入其女王红梅的股票资金账户，据为己有。 
2002年8月18日，被告人王铮利用其担任大连市体彩中心主任、主管财务的职务之便，采取欺骗手段，借给单位副主任王国胜解决住房之机，以兑现“请示”为名，
将本应作为前述第一笔48万元支款根据的“请示”拿出交给财务人员，让出纳员殷淑珍给其开出一张48万元转账支票。后王铮用该款为其女儿王红梅购买个人房产。 
2002年3月至2003年1月，被告人王铮利用担任市体彩中心主任、书记主管财务的职务之便，与市体委竞赛中心（以下简称市竞赛中心）主任孙逢孝签订假租房合同，以支付房租费名义，
套取市体彩中心应上缴省体彩中心的313万元，转存到市竞赛中心在大连商业银行体育场支行开立的支票账户（属账外户）。
并先后安排市体彩中心出纳员殷淑珍和大连金海洋装饰设计工程有限公司（以下简称金海洋公司）出纳员沙晶管理该账户，一直控制该支票账户的银行预留印鉴。
2003年1月14日，王铮利用对该账户的实际控制权，应其朋友于宝军个人入股注册私人公司急需50万元的请托，指使殷从该账户给于开出一张50万元的转账支票，
供于成立金海洋公司验资注册使用。同年1月21日，于将该款返还。 大连市中级人民法院认为：被告人王铮身为国家工作人员，利用职务上的便利，贪污公款93万元，
挪用公款50万元给他人使用，进行营利活动，挪用公款数额巨大，严重地侵犯了国家工作人员职务行为的廉洁性和公款所有权、使用权，其行为已分别构成贪污罪、挪用公款罪。
公诉机关的指控成立。关于王铮及其辩护人所提无罪的辩护意见，经查，王铮以本单位其他有权分房职工的名义进行购房的行为及利用省体彩中心请示领取两套房款，
其中多领的一套房款，均系贪污；王铮从市体彩中心退休后又被返聘，在此期间将市体彩中心存在市竞赛中心账外户上的50万元借给朋友于宝军，作为注册私人公司验资款的行为构成挪用公款罪。
故该辩解和辩护意见缺乏事实和法律依据，本院不予采纳。
但鉴于被告人王铮挪用公款时间较短，可酌情对其从轻处罚。
根据《中华人民共和国刑法》第三百八十二条、第三百八十三条第（一）项、第三百八十四条、第六十九条、第六十四条之规定，判决如下： 
1．被告人王铮犯贪污罪，判处有期徒刑十二年，犯挪用公款罪判处有期徒刑五年，数罪并罚，决定执行有期徒刑十五年。 
2．赃款九十三万元予以追缴，上缴国库。 一审宣判后，王铮向辽宁省高级人民法院提出上诉。王铮上诉提出，45万元是分得的住房款，两个48万元是兑现奖励款，挪用公款时已经退休，不构成犯罪。
其辩护人还提出王铮在审理期间有检举立功表现的辩护意见。 辽宁省高级人民法院经审理后查明： 上诉人王铮在任大连市体育场党总支书记期间，于1997年12月16日，
通过单位会计徐作德以副场长朱可冬名义领取一张人民币45万元的转账支票，在支票的用处栏写明“预收房款”，并于同月17日将该支票存入康泰公司预收款账户。
此后，王铮所在单位进行住房改革，朱可冬自己从单位领取了房款。王铮因在单位未分得住房，便从康泰公司开出购房发票，以其本人购房的名义将该款在单位报销。 
其余事实和证据与一审认定的事实和证据相同，二审予以确认。",
"cpfxgc":"辽宁省高级人民法院认为，王铮虽然冒充朱可冬的名义从单位预支了45万元房款，但其在报销时向单位明确，系其本人购房用款。
由于王铮所在单位进行住房改革，该笔款项为王铮应得款项。王铮及其辩护人所提此笔报销45万元购房款不构成贪污犯罪的上诉理由和辩护意见，予以采纳。
王铮用空军大连房地产处的发票，核销其于2002年8月18日从单位取得人民币48万元的转账支票，系兑现“请示”批准的奖励，不属于贪污。
王铮及其辩护人所提此笔48万元不构成贪污罪的上诉理由和辩护意见，予以采纳。但王铮于2000年6月23日和2000年11月8日虚构装修大连市体彩中心工程的事实，
分别用人民币30万元、18万元假发票2张在大连市体育场改造工程指挥部报销的行为，既不属于真实、合法支出，又与兑现奖励无关，系利用职务上的便利，骗取公共财物的行为，
王铮及其辩护人所提报销该笔装修工程款是兑现“请示”奖励的上诉理由和辩护意见，不予采纳；王铮挪用公款时虽已退休，但其实际上仍然管理、支配着国有财产，具备国家工作人员的身份，
其利用职务之便挪用公款给他人，进行营利活动的行为构成挪用公款罪。王铮及其辩护人所提该行为不构成挪用公款罪的上诉理由和辩护意见，不予采纳。鉴于王铮贪污犯罪的事实发生重大变化，
且贪污人民币48万元的赃款已被追缴；挪用公款时间短，且款项已在案发前返还；其又能检举他人犯罪，经查证属实，构成立功。可对王铮减轻处罚。原判定罪准确，审判程序合法。
根据《中华人民共和国刑事诉讼法》第一百八十九条第（三）项和《中华人民共和国刑法》第三百八十二条、第三百八十三条第（一）项，第三百八十四条第一款、第六十九条、第六十四条、
第六十八条第一款之规定，判决如下：",


'''


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def distribution():
    fw = open('result/distribution_bm25_train.json', 'w', encoding='utf8')
    fq = open('data/phase_2/train/label_top30_dict.json', 'r', encoding='utf8')
    fq = json.load(fq)
    id_ = False
    with open('result/prediction_bm25_train.json', 'r', encoding='utf8') as fb:
        for query_id, candidate_id_list in json.load(fb).items():
            if query_id in fq.keys():
                new_dict = {'qid': query_id} if id_ else [query_id]
                true_id_dict = fq[query_id]
                for candidate_id in candidate_id_list:
                    if str(candidate_id) in true_id_dict.keys():
                        if id_:
                            new_dict[candidate_id] = true_id_dict[str(candidate_id)]
                        else:
                            new_dict.append(true_id_dict[str(candidate_id)])
                    else:
                        if id_:
                            new_dict[candidate_id] = 'None'
                        else:
                            new_dict.append("None")
                fw.write(json.dumps(new_dict, ensure_ascii=False, cls=NpEncoder) + '\n')


def update_label_score():
    from collections import OrderedDict
    f1 = open('result/bm25_scores_train.json', 'r', encoding='utf8')
    score_file = json.load(f1)
    f2 = open('data/phase_2/train_data_win200.json', 'r', encoding='utf8')
    f3 = open('data/phase_2/train/label_top30_dict.json', 'r', encoding='utf8')
    label_file = json.load(f3)
    data = []
    for line in json.load(f2):
        ids = line['labels']
        cand_dict = score_file[str(ids[0])]
        a = np.array(list(cand_dict.values()))
        a = torch.from_numpy(a) / 200
        a = np.array(torch.sigmoid(a))
        rank_id = a.argsort().tolist()[::-1]
        b = np.array(list(cand_dict.keys()))
        OrderDict = OrderedDict()
        for i in rank_id:
            OrderDict[b[i]] = a[i]
        label_rank_id = np.array(list(label_file[ids[0]].values())).argsort().tolist()[::-1]
        c = [list(label_file[ids[0]].keys())[i] for i in label_rank_id[:30]]
        if ids[1] in c:
            line['labels'][2] = OrderDict[ids[1]] + 3*line['labels'][2] if line['labels'][2] != 0 else OrderDict[ids[1]] + 1.0
        else:
            line['labels'][2] = OrderDict[ids[1]]
        data.append(line)

    print(len(data))
    with open('data/phase_2/train_data_win200_plus.json', 'w', encoding='utf8') as fs:
        json.dump(data, fs, ensure_ascii=False, indent=2)


def output_ndcg30():
    from bm25 import cal_ndcg
    f1 = json.load(open('result/prediction_dev.json', 'r', encoding='utf8'))
    fq = json.load(open('data/phase_1/train/label_top30_dict.json', 'r', encoding='utf8'))
    # all_labels = {}
    # for query_id in f1.keys():
    #     all_labels[query_id] = fq[query_id]
    ndcg_30 = cal_ndcg(f1, fq)
    print(ndcg_30)


def count_cpfxgc_length():
    f1 = open('data/case2_train.json', 'r', encoding='utf8')
    data = json.load(f1)
    i = 0
    all_len = []
    for line in data:
        i += 1
        if i in [4428]:
            continue
        length = len(line['candidate'])
        all_len.append(length)
    print(all_len.index(10))
    print('candidate length min: {}'.format(min(all_len)))
    print('candidate length mean: {}'.format(mean(all_len)))
    print('candidate length max: {}'.format(max(all_len)))


def select_dev_samples():
    f1 = open('data/phase_2/train/label_top30_dict.json', 'r', encoding='utf8')
    labels = json.load(f1)
    phase_1_labels = json.load(open('data/phase_1/train/label_top30_dict.json', 'r', encoding='utf8'))
    print(len(labels))
    select_dev_qid_2, select_dev_qid_1 = [], []
    phase_1_qid_2, phase_1_qid_1 = [], []
    length = []
    for qid, line in labels.items():
        length.append(len(list(line.values())))
        if min(list(line.values())) == 2:
            select_dev_qid_2.append(qid)
            if qid in phase_1_labels.keys():
                phase_1_qid_2.append(qid)
        if min(list(line.values())) == 1:
            select_dev_qid_1.append(qid)
            if qid in phase_1_labels.keys():
                phase_1_qid_1.append(qid)
    print(len(select_dev_qid_2))
    print(select_dev_qid_2)
    print(len(select_dev_qid_1))
    print(select_dev_qid_1)
    print(len(phase_1_qid_2))
    print(phase_1_qid_2)
    print(len(phase_1_qid_1))
    print(phase_1_qid_1)
    print(max(length))


def merge_data():
    f1 = open('data/phase_2/train/label_top30_dict.json', 'r', encoding='utf8')
    labels = json.load(f1)
    f1_labels = json.load(open('data/phase_1/train/label_top30_dict.json', 'r', encoding='utf8'))
    print(len(labels))
    select_dev_qid = []
    for qid, line in labels.items():
        if min(list(line.values())) == 2:
            select_dev_qid.append(qid)
        if min(list(line.values())) == 1:
            select_dev_qid.append(qid)
    train_data = json.load(open('data/phase_2/train_data_win200.json', 'r', encoding='utf8'))
    dev_data = json.load(open('data/phase_2/dev_data_win200.json', 'r', encoding='utf8'))
    print(len(train_data))
    print(len(dev_data))
    new_train_data, new_dev_data = [], []
    for line in train_data:
        line['labels'][2] *= 3
        ids = line['labels']
        if ids[0] in select_dev_qid:
            if ids[1] in labels[ids[0]].keys():
                new_train_data.append(line)
        else:
            new_train_data.append(line)
    for line in dev_data:
        line['labels'][2] *= 3
        # ids = line['labels']
        # if ids[0] in select_dev_qid:
        new_dev_data.append(line)
        # else:
        #     new_train_data.append(line)
    print(len(new_train_data))
    print(len(new_dev_data))
    with open('data/phase_2/new_train_data.json', 'w', encoding='utf8') as fs:
        json.dump(new_train_data, fs, ensure_ascii=False, indent=2)
    with open('data/phase_2/new_dev_data.json', 'w', encoding='utf8') as fd:
        json.dump(new_dev_data, fd, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # output_ndcg30()
    # distribution()
    # update_label_score()
    # count_cpfxgc_length()
    # select_dev_samples()
    merge_data()


# with open('data/case2_train.json', 'r', encoding='utf8') as f1:
#     a, b = [], []
#     i = 0
#     for line in json.load(f1):
#         # i += 1
#         # if i < 1358:
#         #     continue
#         q_length = len(line['query'])
#         c_length = len(line['candidate'])
#         a.append(q_length)
#         b.append(c_length)
# print(len(a))
# print(b.index(934))
# print('query number min: {}'.format(min(a)))
# print('query number mean: {}'.format(mean(a)))
# print('query number max: {}'.format(max(a)))
#
# print('candidate number min: {}'.format(min(b)))
# print('candidate number mean: {}'.format(mean(b)))
# print('candidate number max: {}'.format(max(b)))

# train_npy = np.load('data/large_extract.npy')[:, :101, :]
# dev_npy = np.load('data/small_extract.npy')
# new_npy = np.concatenate((train_npy, dev_npy), axis=0)
# np.save('data/large_extract_all', new_npy)


# ranks = [1, 3, 0, 2, 1, 0]
# new_ranks = [i - 2 if i > 1 else i for i in ranks]
# print(new_ranks)
# with open('data/query_convert/case_dev.json', 'r', encoding='utf8') as fa:
#     data = json.load(fa)
#     for i in data:
#         ori_label = i['label'][2] * 3
#         if ori_label > 1:
#             i['label'][2] = int(ori_label - 2)
#         else:
#             i['label'][2] = int(ori_label)
#         update_label.append(i)
# with open('data/extract_all.json', 'w', encoding='utf8') as fs:
#     json.dump(c, fs, ensure_ascii=False, cls=NpEncoder)


#     for i in data:
#         length = len(i['query'])
#         length_ = len(i['candidate'])
#         all_len.append(length)
#         all_candi_len.append(length_)
#
# print('query length min: {}'.format(min(all_len)))
# print('query length mean: {}'.format(mean(all_len)))
# print('query length max: {}'.format(max(all_len)))
#
# print('candidate length min: {}'.format(min(all_candi_len)))
# print('candidate length mean: {}'.format(mean(all_candi_len)))
# print('candidate length max: {}'.format(max(all_candi_len)))


# all_labels = []
# all_text_len = []
# all_sent_len = []
# all_labels_len = []
# with open('./data/extract_large.json', 'r', encoding='utf8') as ff:
#     for line in json.load(ff):
#         labels = line['labels']
#         leng = len(line['ajjbqk'])
#         for sent in line['ajjbqk']:
#             sent_len = len(sent)
#             all_sent_len.append(sent_len)
#         all_text_len.append(leng)
#         all_labels.extend(labels)
#         all_labels_len.append(len(labels))
#
# print('Labels number min: {}'.format(min(all_labels)))
# print('Labels number mean: {}'.format(mean(all_labels)))
# print('Labels number max: {}'.format(max(all_labels)))
#
# print('Texts length min: {}'.format(min(all_text_len)))
# print('Texts length mean: {}'.format(mean(all_text_len)))
# print('Texts length max: {}'.format(max(all_text_len)))
#
# print('Sentence length min: {}'.format(min(all_sent_len)))
# print('Sentence length mean: {}'.format(mean(all_sent_len)))
# print('Sentence length max: {}'.format(max(all_sent_len)))
#
# print('Labels length min: {}'.format(min(all_labels_len)))
# print('Labels length mean: {}'.format(mean(all_labels_len)))
# print('Labels length max: {}'.format(max(all_labels_len)))