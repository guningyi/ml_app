import nltk
import random

def makePairs(arr):
    pairs = []
    for i in range(len(arr)):
        if i < len(arr)-1:
            temp = (arr[i], arr[i+1])
            pairs.append(temp)
    return pairs

def generate(cfd, word = 'the', num = 50):
    for i in range(num):
        arr = []                 # make an array with the words shown by proper count
        for j in cfd[word]:
            for k in range(cfd[word][j]):
                arr.append(j)

        #print(word, end=' ')
        print word
        word = arr[int((len(arr))*random.random())]

if __name__ == "__main__":
    print "start process!"
    file = open('C:/Users/eniiguu/Desktop/message.txt', 'r')
    walden = file.read()
    walden = walden.split()
    pairs = makePairs(walden)
    cfd = nltk.ConditionalFreqDist(pairs)
    generate(cfd)