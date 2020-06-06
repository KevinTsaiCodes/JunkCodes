import tensorflow as tf
import numpy as np

modle = __import__("7-14  MKR")
MKR = modle.MKR

def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    kg = data[7]
    #定義MKR模型
    model = MKR(args, n_user, n_item, n_entity, n_relation)

    #設定評估模型的參數
    user_num = 100 #選100個使用者
    k_list = [1, 2, 5, 10, 20, 50, 100]#為每個使用者推薦指定個數的電影
    train_record = get_user_record(train_data, True)#獲得該使用者關聯的電影資料
    test_record = get_user_record(test_data, False)#獲得該使用者喜歡的電影資料
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num: #控制測試使用者個數為100
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))#以集合模式產生電影的id

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epochs):
            #訓練RS模型
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:
                _, loss = model.train_rs(sess, get_feed_dict_for_rs(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(loss)

            #訓練KGE模型
            if step % args.kge_interval == 0:
                np.random.shuffle(kg)
                start = 0
                while start < kg.shape[0]:
                    _, rmse = model.train_kge(sess, get_feed_dict_for_kge(model, kg, start, start + args.batch_size))
                    start += args.batch_size
                    if show_loss:
                        print(rmse)

            #輸出本次的訓練結果
            train_auc, train_acc = model.eval(sess, get_feed_dict_for_rs(model, train_data, 0, train_data.shape[0]))
            eval_auc, eval_acc = model.eval(sess, get_feed_dict_for_rs(model, eval_data, 0, eval_data.shape[0]))
            test_auc, test_acc = model.eval(sess, get_feed_dict_for_rs(model, test_data, 0, test_data.shape[0]))
            print('\nepoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                  % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))

            #計算top-K的評估結果
            if show_topk:
                precision, recall, f1 = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list)
                print('precision: ', end='')  #輸出精確率
                for i in precision:
                    print('%.4f' % i, end=' ')
                print('\nrecall:'.ljust(len('precision: ')+1), end='')#輸出召回率
                for i in recall:
                    print('%.4f' % i, end=' ')
                print('\nf1:'.ljust(len('precision: ')+1), end='')#輸出綜合結果
                for i in f1:
                    print('%.4f' % i, end=' ')
#定義函數，用於植入rs模型
def get_feed_dict_for_rs(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2],
                 model.head_indices: data[start:end, 1]}
    return feed_dict

#定義函數，用於植入kge模型
def get_feed_dict_for_kge(model, kg, start, end):
    feed_dict = {model.item_indices: kg[start:end, 0],
                 model.head_indices: kg[start:end, 0],
                 model.relation_indices: kg[start:end, 1],
                 model.tail_indices: kg[start:end, 2]}
    return feed_dict

#計算MKR模型最終評估結果
def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list: #為每個使用者推薦電影
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        #獲得所有的電影id及推薦分數
        items, scores = model.get_scores(sess, {model.user_indices: [user] * len(test_item_list),
                                                model.item_indices: test_item_list,
                                                model.head_indices: test_item_list})
        #對推薦電影的分數進行排序
        for item, score in zip(items, scores):
            item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        #當推薦個數為[1, 2, 5, 10, 20, 50, 100]時，分別計算模型的精確率與召回率
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]  #計算精確率的平均值
    recall = [np.mean(recall_list[k]) for k in k_list]        #計算召回率的平均值
    f1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(len(k_list))] #綜合評分值
    return precision, recall, f1

#定義函數，根據設定傳回與使用者關聯的電影資料（內含喜歡的和不喜歡的）。
def get_user_record(data, is_train):#若果is_train是False，則只傳回使用者喜歡的電影資料
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

if __name__ == '__main__':
     tf.reset_default_graph()
     import argparse
     data_loader = __import__("7-16  data_loader")
     load_data = data_loader.load_data
     np.random.seed(555)
     parser = argparse.ArgumentParser()
     #預設使用movie資料集
     parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
     parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
     parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
     parser.add_argument('--L', type=int, default=1, help='number of low layers')
     parser.add_argument('--H', type=int, default=1, help='number of high layers')
     parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
     parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
     parser.add_argument('--lr_rs', type=float, default=0.02, help='learning rate of RS task')
     parser.add_argument('--lr_kge', type=float, default=0.01, help='learning rate of KGE task')
     parser.add_argument('--kge_interval', type=int, default=3, help='training interval of KGE task')
     show_loss = False
     show_topk = False

     args = parser.parse_args()
     data = load_data(args)#載入資料集
     train(args, data, show_loss, show_topk) #進行訓練及評估


