 
from scipy.io import arff
from io import StringIO
import pandas as pd
import math
from pandas.api.types import is_numeric_dtype



def main():
    data = arff.loadarff('adult-big.arff')
    df = pd.DataFrame(data[0])
    df = df.sample(frac=1).reset_index(drop=True)
    #should shuffle before doing the cross validation
    df1 = df #4884 0-4883 -> you only need one copy and you do all the changes to that
    #print(df)
    
    for i in range(1,2): #change this back to 10 later
        getAverage(i, df1)
        df1 = df

def entropy(size, part1, part2):
    if part1 == 0:
        ent = -1*(part2*math.log2(part2))
    elif part2 == 0:
        ent = -1*(part1*math.log2(part1))
    else:
        ent = (-1*((part1*math.log2(part1))+(part2*math.log2(part2))))
    return ent

def discretizeRec(df, splits, thelist):
    if splits > 8:#design decision
        return
    #do discretization for this fold or whatever
    totalsize = len(df.index)
    above = df.loc[df['class'] == b'>50K']
    below = df.loc[df['class'] == b'<=50K']
    p1 = len(above.index)/totalsize
    p2 = len(below.index)/totalsize
   
    initialent = entropy(totalsize, p1, p2)
    currentmin = 0
    split = 0
    min = int(df['age'].min())
    max = int(df['age'].max())

    for i in range (min, max): 
        #need to know num in bin, num >50 and num <=50
        #calculate entropy of 0-i
        top = df.loc[df['age']<=i]
        bottom = df.loc[df['age']>i]
        topbelow = top.loc[top['class'] == b'<=50K']
        topabove = top.loc[top['class'] == b'>50K']
        bbelow = bottom.loc[bottom['class'] == b'<=50K']
        babove = bottom.loc[bottom['class'] == b'>50K']

        topsize = len(top.index)
        botsize = len(bottom.index)
        top1 = len(topabove.index)/topsize
        top2 = len(topbelow.index)/topsize
        bot1 = len(babove.index)/botsize
        bot2 = len(bbelow.index)/botsize
        
        enttop = entropy(topsize, top1, top2) 
        entbottom = entropy(botsize, bot1, bot2)
        
        netent = (topsize/totalsize)*enttop + (botsize/totalsize)*entbottom
        print("{}, {}".format(i, netent))

        if initialent - netent > currentmin:
            currentmin = initialent-netent
            split = i
            dfinal1 = top
            dfinal2 = bottom

        #calculate entropy of i+1-90

    print(currentmin)
    print(split)
    thelist.append(split)
    splits += 2
    print("DONE WITH ONE")
    discretizeRec(dfinal1, splits, thelist)
    discretizeRec(dfinal2, splits, thelist)


    #call on top of split, bottom of split, increase split by 2
    #return a list of the values where we're splitting

def discretize(dftrain, dftest):
    dftrain.sort_values(by='age', inplace= True)
    #print(dftrain['age'])
    dftrain.to_csv('ordered_stuff.csv', index=False) 

    splitPoints = []
    discretizeRec(dftrain, 2, splitPoints)
    print(splitPoints)
    #here we use the list that we've generated and we ... discretize
    


    

def getAverage(testSeg, datfram):
    #check to see which segment is the test segment, it won't be used in creating averages
    #DO NOT HARDCODE THE NUMBER
    if testSeg == 1:
        dftest = datfram.iloc[:4884, :]
        dftrain = datfram.iloc[4884:, :]

    elif testSeg > 1 and testSeg < 10:
        last = testSeg-1
        dftest = datfram.iloc[4884*last:4884*testSeg, :]
        dftrain1 = datfram.iloc[:4884*last, :]
        dftrain2 = datfram.iloc[4884*testSeg:]
        frames = [dftrain1, dftrain2]
        dftrain = pd.concat(frames)
    else:
        dftest = datfram.iloc[4884*9:, :]
        dftrain = datfram.iloc[:4884*9, :]


    #split test data to get averages
    dfover50 = dftrain.loc[dftrain['class']== b'>50K']
    dfunder50 = dftrain[dftrain['class']== b'<=50K']
    #print(dfover50)
    for col in dfover50.columns:
        av = calcAverage(col, dfover50) #get the average for that column
        dfover50[col].replace(b'?', av, inplace=True)
    for col in dfunder50.columns: 
        av = calcAverage(col, dfunder50) #get the average for that column
        dfunder50[col].replace(b'?', av, inplace=True)

    for col in dftest:
        if len(dfover50.columns) > len(dfunder50.columns):
            av = calcAverage(col, dfover50)
        else:
            av = calcAverage(col, dfunder50)
        dftest[col].replace(b'?', av, inplace=True)
    
    #print("DF OVER 50")
    #print(dfover50)
    #print("DF UNDER 50")
    #print(dfunder50)
    #print("test")
    #print(dftest)

    dfFinalTrain = pd.concat([dfover50, dfunder50])
    discretize(dfFinalTrain, dftest)
    #here need to use the bins fr



def calcAverage(columName, df):
    if is_numeric_dtype(df[columName]):
        return round(df[columName].mean())
    else:
        return df[columName].mode()[0]


if __name__ == "__main__":
    main()