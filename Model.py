# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import argparse
from DataSet import DataSet
import sys
import heapq
import math

def main():
    parser = argparse.ArgumentParser(description="Options")
    
    parser.add_argument('-dataName', action='store', dest='dataName', default='miRNA-disease')
    parser.add_argument('-negNum', action='store', dest='negNum', default=1, type=int)
    parser.add_argument('-userLayer', action='store', dest='userLayer', default=[32,16])
    parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[32,16])
    parser.add_argument('-userFLayer', action='store', dest='userFLayer', default=[32,16])
#     parser.add_argument('-itemFLayer', action='store', dest='itemFLayer', default=[128, 32])
    parser.add_argument('-reg', action='store', dest='reg', default=0.5)
    parser.add_argument('-alfha', action='store',dest='alfha',default = 0.7)
    parser.add_argument('-lr', action='store', dest='lr', default=0.00001)
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=400, type=int)
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=5000, type=int)
    parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=20)
#     parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')
    parser.add_argument('-topK', action='store', dest='topK', default=10)
    parser.add_argument('-cvfold', action='store', dest='cvfold', default=1)

    args = parser.parse_args()

    #generate 5-fold training list
    training_samples_num = 11252

    fold_num = args.cvfold
    l = np.arange(training_samples_num)
    np.random.shuffle(l)

    Testinglist = []             

    classifier = Model(args,Testinglist)
    s_ItemUser, s_UserF, s_UserEmbedding  = classifier.run()
    np.savetxt('s_LMI.txt',s_ItemUser)
    np.savetxt('s_MDA.txt',s_UserF)
    np.savetxt('s_MicroRNA_Embedding.txt',s_UserEmbedding)
           
class Model:
    def __init__(self, args, Testinglist):
        self.dataName = args.dataName
        self.dataSet = DataSet(self.dataName, Testinglist)
        self.shape = self.dataSet.shape
        self.maxRate = self.dataSet.maxRate
        
        self.userFlen = self.dataSet.UF_len
        self.allMatrix = self.dataSet.getTrainAll()
        
        self.train = self.dataSet.train
        self.test = self.dataSet.test

        self.negNum = args.negNum
        self.reg = args.reg
        self.alfha = args.alfha
        self.testNeg = self.dataSet.getTestNeg(self.test, 20)
        
        self.cvfold_num = args.cvfold
        
        self.add_embedding_matrix()
        print("add_embedding_matrix SUCCESS")
        self.add_placeholders()
        print("add_placeholders SUCCESS")

        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        self.userFLayer = args.userFLayer
        self.add_model()
        print("add_model SUCCESS")

        self.add_loss()
        print("add_loss SUCCESS")

        self.lr = args.lr
        self.add_train_step()
        print("add_train_step SUCCESS")

#         self.checkPoint = args.checkPoint
        self.init_sess()
        print("init_sess SUCCESS")

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize

        self.topK = args.topK
        self.earlyStop = args.earlyStop
        
    def add_placeholders(self):
        self.user = tf.placeholder(tf.int32)
        self.item = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.float32)

    def add_embedding_matrix(self):
        self.user_item_embedding = tf.convert_to_tensor(self.dataSet.getEmbedding())
        self.item_user_embedding = tf.transpose(self.user_item_embedding)
        user_feature = self.dataSet.getFeatureEmbedding()
        self.user_feature_embedding = tf.convert_to_tensor(user_feature)
        self.feature_user_embedding = tf.transpose(tf.convert_to_tensor(user_feature))

    def add_model(self):
        user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
        item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)
        Fea_user_input = self.feature_user_embedding
        userFea_input = tf.nn.embedding_lookup(self.user_feature_embedding, self.user) 

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.userLayer)-1):
                W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
                b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            item_out = tf.matmul(item_input, item_W1)
            for i in range(0, len(self.itemLayer)-1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
                b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

        with tf.name_scope("UserFeature_Layer"):
            userF_W1 = init_variable([self.shape[0], self.userFLayer[0]], "userF_W1")
            userF_out = tf.matmul(Fea_user_input, userF_W1)
            for i in range(0, len(self.userFLayer)-1):
                W = init_variable([self.userFLayer[i], self.userFLayer[i+1]], "userF_W"+str(i+2))
                b = init_variable([self.userFLayer[i+1]], "userF_b"+str(i+2))
                userF_out = tf.nn.relu(tf.add(tf.matmul(userF_out, W), b))                
        
        norm_user_output = tf.norm(user_out, axis=1)
        norm_item_output = tf.norm(item_out, axis=1)        
        norm_userF_output = tf.norm(userF_out)
        
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keep_dims=False) / (norm_item_output* norm_user_output)        
               
        self.user_out = user_out
        self.yu_ = tf.matmul(userF_out, tf.transpose(user_out)) / (norm_user_output* norm_userF_output)             
        
        self.y_ = tf.maximum(1e-6, self.y_)
        self.yu_ = tf.maximum(1e-6, self.yu_)
                
        self.y_ = tf.minimum(1-1e-6, self.y_)
        self.yu_ = tf.minimum(1-1e-6, self.yu_)
                
        self.yu_temp = tf.subtract(self.yu_, tf.transpose(userFea_input))

    def add_loss(self):
        regRate = self.rate / self.maxRate
             
        losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
        loss = -tf.reduce_sum(losses)

        #MSE
        losses_u = tf.reduce_mean(tf.square(self.yu_temp)) / self.dataSet.UF_len
        t_vars = tf.trainable_variables()
        self.loss = loss * self.alfha + self.reg * tf.add_n([tf.nn.l2_loss(v) for v in t_vars if v.name.startswith('Item_Layer')]) + 0.5 * tf.add_n([tf.nn.l2_loss(v) for v in t_vars if v.name.startswith('User_Layer')])
        self.loss_u = losses_u * (1-self.alfha) + self.reg * tf.add_n([tf.nn.l2_loss(v) for v in t_vars if v.name.startswith('UserFeature_Layer')]) + 0.5 * tf.add_n([tf.nn.l2_loss(v) for v in t_vars if v.name.startswith('User_Layer')])
    def add_train_step(self):
        '''
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.lr, global_step,
                                             self.decay_steps, self.decay_rate, staircase=True)
        '''
        t_vars = tf.trainable_variables()
        var_list1 = []
        var_list2 = []
        for v in t_vars:
            if v.name.startswith('Item_Layer') or v.name.startswith('User_Layer'):
                var_list2.append(v)
        for v in t_vars:
            if v.name.startswith('UserFeature_Layer') or v.name.startswith('User_Layer'):
                var_list1.append(v)
        
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step1 = optimizer.minimize(self.loss_u, var_list = var_list1)
        self.train_step2 = optimizer.minimize(self.loss, var_list = var_list2)

    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())


    def run(self):
        loss_epoch = []
        loss_u_epoch = []
        print("Start Training!")
        for epoch in range(self.maxEpochs):
            print("="*20+"Epoch ", epoch, "="*20)
            loss, loss_u = self.run_epoch(self.sess)
            loss_epoch.append(loss)
            loss_u_epoch.append(loss_u)
            print('='*50)
        s_ItemUser, s_UserF, s_UserEmbedding = self.get_testScore(self.sess)
        return s_ItemUser, s_UserF, s_UserEmbedding

    def run_epoch(self, sess, verbose=10):
        train_u, train_i, train_r = self.dataSet.getInstances(self.train, self.negNum)
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1

        losses = []
        losses_u = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i+1)*self.batchSize])
            train_u_batch = train_u[min_idx: max_idx]
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]

            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            _, _, tmp_loss, tmp_loss_u = sess.run([self.train_step1, self.train_step2, self.loss, self.loss_u], feed_dict=feed_dict)
#             _, tmp_loss, tmp_loss_u = sess.run([self.train_step2, self.loss, self.loss_u], feed_dict=feed_dict)
            losses.append(tmp_loss)
            losses_u.append(tmp_loss_u)
            if verbose and i % verbose == 0:
                if np.isnan(np.mean(losses[-verbose:])):
                    raise ValueError                                
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, num_batches, np.mean(losses[-verbose:])
                ))
                sys.stdout.flush()
        loss = np.mean(losses)
        loss_u = np.mean(losses_u)
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss, loss_u

    def create_feed_dict(self, u, i, r=None):
        return {self.user: u,
                self.item: i,
                self.rate: r,}

    def evaluate(self, sess, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0

        hr =[]
        NDCG = []
        testUser = self.testNeg[0]
        testItem = self.testNeg[1]
        for i in range(len(testUser)):
            target = testItem[i][0]
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = sess.run(self.y_, feed_dict=feed_dict)
            item_score_dict = {}

            for j in range(len(testItem[i])):
                item = testItem[i][j]
                item_score_dict[item] = predict[j]

            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        return np.mean(hr), np.mean(NDCG)
    
    def get_testScore(self, sess):
        train_allu = self.allMatrix[0]
        train_alli = self.allMatrix[1]
        train_allr = self.allMatrix[2]
        feed_dict = self.create_feed_dict(train_allu, train_alli, train_allr)
        predict_p, predict_q, predict_user_out = sess.run([self.y_, self.yu_, self.user_out], feed_dict=feed_dict)
        
        
        s_ItemUser = np.zeros((799, 268))
        for i in range(len(predict_p)):
            s_ItemUser[train_alli[i],train_allu[i]] = predict_p[i]
        
        
        print(predict_user_out.shape)
        s_UserEmbedding = np.zeros((268, predict_user_out.shape[1]))
         #to get the score of userFeature matrix        
        s_UserF = np.zeros((541, 268))
        u_set = set()
        for i in range(predict_q.shape[1]):
            if train_allu[i] not in u_set:
                s_UserF[:,train_allu[i]] = predict_q[:, i]
                s_UserEmbedding[train_allu[i],:] = predict_user_out[i, :]
                u_set.add(train_allu[i])
        
        return s_ItemUser, s_UserF, s_UserEmbedding

if __name__ == '__main__':
    main()
