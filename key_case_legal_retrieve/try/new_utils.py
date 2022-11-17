import numpy as np
import six


def parallel_apply(
    func,
    iterable,
    workers,
    max_queue_size,
    callback=None,
    dummy=False,
    random_seeds=True
):
    """多进程或多线程地将func应用到iterable的每个元素中。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是
    输出可能是func(c), func(a), func(b)。
    参数：
        callback: 处理单个输出的回调函数；
        dummy: False是多进程/线性，True则是多线程/线性；
        random_seeds: 每个进程的随机种子。
    """
    if dummy:
        from multiprocessing.dummy import Pool, Queue
    else:
        from multiprocessing import Pool, Queue

    in_queue, out_queue, seed_queue = Queue(max_queue_size), Queue(), Queue()
    if random_seeds is True:
        random_seeds = [None] * workers
    elif random_seeds is None or random_seeds is False:
        random_seeds = []
    for seed in random_seeds:
        seed_queue.put(seed)

    def worker_step(in_queue, out_queue):
        """单步函数包装成循环执行
        """
        if not seed_queue.empty():
            np.random.seed(seed_queue.get())
        while True:
            i, d = in_queue.get()
            r = func(d)
            out_queue.put((i, r))

    # 启动多进程/线程
    pool = Pool(workers, worker_step, (in_queue, out_queue))

    if callback is None:
        results = []

    # 后处理函数
    def process_out_queue():
        out_count = 0
        for _ in range(out_queue.qsize()):
            i, d = out_queue.get()
            out_count += 1
            if callback is None:
                results.append((i, d))
            else:
                callback(d)
        return out_count

    # 存入数据，取出结果
    in_count, out_count = 0, 0
    for i, d in enumerate(iterable):
        in_count += 1
        while True:
            try:
                in_queue.put((i, d), block=False)
                break
            except six.moves.queue.Full:
                out_count += process_out_queue()
        if in_count % max_queue_size == 0:
            out_count += process_out_queue()

    while out_count != in_count:
        out_count += process_out_queue()

    pool.terminate()

    if callback is None:
        results = sorted(results, key=lambda r: r[0])
        return [r[1] for r in results]


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


def jbqk_split(ajjbqk, query, select=3):
    ajjbqk = string_rule(ajjbqk)
    ajjbqk_list = list(set([s for s in re.split(r'[。，：；]', ajjbqk) if len(s) > 0]))
    corpus = []
    for comma_sent in ajjbqk_list:
        words_list = [w for w in jieba.lcut(comma_sent) if w not in stop_words]
        corpus.append(words_list)
    bm25model = bm25.BM25(corpus)
    q_tokens = [w for w in jieba.lcut(query) if w not in stop_words]
    scores = bm25model.get_scores(q_tokens)
    rank_index = np.array(scores).argsort().tolist()[::-1]
    new_jbqk = [ajjbqk_list[i] for i in rank_index]
    # new_jbqk_ = '。'.join(new_jbqk)
    scores_list = np.array([scores[i] for i in rank_index]).reshape([-1, 1])
    min_max_scaler = preprocessing.MinMaxScaler()
    scores_norm = min_max_scaler.fit_transform(scores_list)
    return new_jbqk, scores_norm[:, 0].tolist()


def test():
    import torch
    import json, re
    from numpy import mean
    print(torch.cuda.is_available())

    # D = []
    # with open('data/textrank_filter/dev.json', 'r', encoding='utf8') as f:
    #     for line in f:
    #         D.append(json.loads(line))
    # with open('data/textrank_filter/filter_dev.json', 'w', encoding='utf8') as f1:
    #     json.dump(D, f1, ensure_ascii=False, indent=2)

    f1 = open('data/key_train.json', 'r', encoding='utf8')
    f2 = open('data/key_dev.json', 'r', encoding='utf8')
    data1 = json.load(f1)
    data2 = json.load(f2)
    data1.extend(data2)

    query_len_list, candi_len_list = [], []
    i = 0
    sent_num_list = []
    for pair in data1:
        i += 1
        # if i == 6105:
        #     sent = pair['query']
        #     print('label:{}'.format(pair['label']))
        #     sent_list = sent.split('。')
        #     print('origin list length: {}'.format(len(sent_list)))
        #     print('After processing: {}'.format(len(set(sent_list))))
        #     c = '。'.join(sent_list)
        #     print(c)
        #     print('length: {}'.format(len(c)))
        #     break
        sent_list = [s for s in re.split(r'[。！？]', pair['query']) if len(s) > 0]
        sent_num_list.append(len(sent_list))
        len_query, len_candi = len(pair['query']), len(pair['candidate'])
        query_len_list.append(len_query)
        candi_len_list.append(len_candi)

    print('query sentence number min: {}'.format(min(sent_num_list)))
    print('query sentence number mean: {}'.format(mean(sent_num_list)))
    print('query sentence number max: {}'.format(max(sent_num_list)))
    print('query max length: {}'.format(max(query_len_list)))
    print('query min length: {}'.format(min(query_len_list)))
    print('query average length: {}'.format(mean(query_len_list)))
    print('candidate max length: {}'.format(max(candi_len_list)))
    print('candidate min length: {}'.format(min(candi_len_list)))
    print('candidate average length: {}'.format(mean(candi_len_list)))
    # print(candi_len_list.index(1396))
    print(query_len_list.index(1199))
    # with open('data/all_data.json', 'w', encoding='utf8') as f3:
    #     json.dump(data1, f3, ensure_ascii=False, indent=2)