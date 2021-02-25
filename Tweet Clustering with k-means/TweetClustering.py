import random
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

class TweetClustering:
    def preProcess(self, df):
        df = df.drop([0, 1, 2, 3, 4], axis=1)
        df[5] = df[5].str[5:]
        df[5] = df[5].apply(lambda x: re.sub("http(.*)", "", x)) \
            .apply(lambda x: re.sub("^@(.*)", "(.*)", x)) \
            .apply(lambda x: re.sub("#", "", x)) \
            .apply(lambda x: x.lower())
        data = []
        for each in df[5]:
            wordsList = each.split()
            wordsSet = set()
            for word in wordsList:
                wordsSet.add(word)
            data.append(wordsSet)
        return data

    def initalizeRandomCenters(self, k, data):
        start, end = 0, len(data) - 1
        randomCenters = []
        indexSet = set([])
        iterator = 0
        while iterator < k:
            randCenter = random.randint(start, end)
            if randCenter not in indexSet:
                iterator += 1
                indexSet.add(randCenter)
                randomCenters.append(data[randCenter])
        return randomCenters

    def calculateJaccardDistance(self, a, b):
        x, y = len(a.intersection(b)), len(a.union(b))
        if x == 0:
            return 1
        return 1 - (float((x / y)))

    def assignCluster(self, tweets, centers):
        clusters = [None] * (len(centers))
        for t in tweets:
            index = 0
            mindist = 2
            for c in centers:
                dist = self.calculateJaccardDistance(t, c)
                if dist < mindist:
                    mindist = dist
                    index = centers.index(c)
                elif dist == mindist:
                    index = random.choice([index, centers.index(c)])
                    index = centers.index(c)
            if clusters[index] == None:
                cluster = []

            else:
                cluster = clusters[index]
            cluster.append(tweetData[tweets.index(t)])
            clusters[index] = cluster
        return clusters

    def updateClusterCenter(self, tweet):
        center = 0
        maxDist = sys.maxsize
        for t in tweet:
            dist = 0
            for each in tweet:
                dist += self.calculateJaccardDistance(t, each)
            if dist < maxDist:
                center = tweet.index(t)
                maxDist = dist
            elif dist == maxDist:
                center = random.choice([center, tweet.index(t)])
        return center

    def calculateSSE(self, clusters, centroids):
        error = 0
        for c in clusters:
            error = 0
            for t in c:
                error += (self.calculateJaccardDistance(centroids[clusters.index(c)], t)) ** 2
            error *= 2
        return error


if __name__ == "__main__":
    dataSet = 'cbchealth.txt'
    data = pd.read_fwf(filepath_or_buffer=dataSet, sep='\t', header=None)
    tweetsClustering = TweetClustering()
    tweetData = tweetsClustering.preProcess(data)
    print("Number of Tweets:" + str(len(tweetData)))
    # klist = [1,3,5,10,15,18,25,38,50,67,79,86,100,111,125,135]
    # sseList=[]
    # stepsList=[]
    # results=[]
    klist = [99]
    for k in klist:
        # row={}
        print("k : ", k)
        # row['k']=k
        centers = tweetsClustering.initalizeRandomCenters(k, tweetData)
        converged = False
        steps = 0
        while converged == False:
            steps += 1
            newCluster = tweetsClustering.assignCluster(tweetData, centers)
            newCenters = []
            for value in newCluster:
                center = tweetsClustering.updateClusterCenter(value)
                newCenters.append(value[center])
            if centers == newCenters:
                converged = True
            else:
                centers = newCenters

        resultCluster = newCluster
        finalCenters = newCenters

        sse = tweetsClustering.calculateSSE(resultCluster, newCenters)
        print("Steps to converge :,", steps)
        # row['Steps'] = steps
        # row['SSE'] = sse
        # stepsList.append(steps)
        # sseList.append(sse)
        print("SSE : ", sse)
        print("clusters size: ")
        tweetClusterData = ""
        for value in resultCluster:
            tweetClusterData += str(resultCluster.index(value) + 1) + ": " + str(len(value)) + " tweets \n"
        print(tweetClusterData)
        # row['Cluster Size'] = tweetClusterData
        # results.append(row)
    # df = pd.DataFrame(results)
    # df.to_excel("observations.xlsx")
    # plt.plot(klist, sseList)
    # plt.xlabel('K ')
    # plt.ylabel('Sum Square Error ')
    # plt.title('K vs SSE')
    # plt.show()
    # plt.plot(klist, stepsList)
    # plt.xlabel('K ')
    # plt.ylabel('Steps to Converge ')
    # plt.title('K vs Steps to Converge')
    # plt.show()
