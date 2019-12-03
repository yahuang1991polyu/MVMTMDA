# -*- Encoding:UTF-8 -*-

import numpy as np
import sys


class DataSet(object):
    def __init__(self, fileName, Testinglist):
        self.data, self.shape = self.getData(fileName)
        self.userdata, self.UF_len = self.getFeatureData()
        self.train, self.test = self.getTrainTest(Testinglist)
        self.trainDict = self.getTrainDict()
        self.testDict = self.getTestDict()

    def getData(self, fileName):
        if fileName == 'miRNA-disease':
            print("Loading lncRNA-miRNA data set...")
            data = []
            filePath = './Data/miRNA-disease_id.txt'
#             filePath = './Data/ml-1m/ratings.dat'
            u = 0
            i = 0
            maxr = 0.0
            with open(filePath, 'r') as f:
                for line in f:
                    if line:
                        lines = line[:-1].split("::")
                        user = int(lines[0])
                        movie = int(lines[1])
                        score = float(lines[2])
                        data.append((user, movie, score))
                        if user > u:
                            u = user
                        if movie > i:
                            i = movie
                        if score > maxr:
                            maxr = score
            self.maxRate = maxr
            print("Loading Success!\n"
                  "Data Info:\n"
                  "\tUser Num: {}\n"
                  "\tItem Num: {}\n"
                  "\tData Size: {}".format(u, i, len(data)))
            return data, [u, i]
        else:
            print("Current data set is not support!")
            sys.exit()
    
    def getFeatureData(self):
        UserFeature_data = []
        filePath_U = './Data/lncRNA-miRNA_id.txt'
        UF_len = 0
        with open(filePath_U,'r') as f_u:
            for line in f_u:
                if line:
                    lines = line[:-1].split("::")
                    user = int(lines[1])
                    user_f = int(lines[0])
                    score = float(lines[2])
                    UserFeature_data.append((user,user_f,score))
                    if user_f > UF_len:
                        UF_len = user_f
        print("Loading Feature Success! \n"
                  "\tUser feature length: {}\n".format(UF_len))
        return UserFeature_data, UF_len

    def getTrainTest(self, Testinglist):
        data = self.data
        train = []
        test = []
        for i in range(len(data)):             
            user = data[i][0]-1
            item = data[i][1]-1
            rate = data[i][2]
            if i in Testinglist:
                test.append((user, item, rate))            
            else:
                train.append((user, item, rate))        
        return train, test
    

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict
    
    def getTestDict(self):
        dataDict = {}
        for i in self.test:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getEmbedding(self):
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating
        return np.array(train_matrix)
    
    def getFeatureEmbedding(self):
        UserFeature_matrix = np.zeros([self.shape[0], self.UF_len], dtype=np.float32) 
        
        for i in self.userdata:
            user = i[0]-1
            u_feature = i[1]-1
            rating = i[2]
            UserFeature_matrix[user][u_feature] = rating
            
        
        return np.array(UserFeature_matrix)

    def getInstances(self, data, negNum):
        user = []
        item = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)
    
    
    #getTrainAll: return training data along with other candidates
    def getTrainAll(self):
        TrainDict = self.getTrainDict()
        user = []
        item = []
        rate = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                user.append(i)
                item.append(j)
                if (i,j) in TrainDict:
                    rate.append(1)
                else:
                    rate.append(0)
        return np.array(user), np.array(item), np.array(rate)

    #negNum: number of negative samples for each user
    def getTestNeg(self, testData, negNum):
        user = []
        item = []
        for s in testData:
            tmp_user = []
            tmp_item = []
            u = s[0]
            i = s[1]
            tmp_user.append(u)
            tmp_item.append(i)
            neglist = set()
            neglist.add(i)
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (u, j) in self.trainDict or j in neglist:
                    j = np.random.randint(self.shape[1])
                neglist.add(j)
                tmp_user.append(u)
                tmp_item.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
        return [np.array(user), np.array(item)]
