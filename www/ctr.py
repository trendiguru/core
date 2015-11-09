from math import sqrt
import matplotlib.pyplot as plt

def confidence(clicks, impressions):
    n = impressions
    if n == 0: return 0
    z = 1.6 #1.6 -> 95% confidence
    phat = float(clicks) / n
    denorm = 1. + (z*z/n)
    enum1 = phat + z*z/(2*n)
    enum2 = z * sqrt(phat*(1-phat)/n + z*z/(4*n*n))
    return (enum1-enum2)/denorm, (enum1+enum2)/denorm

def wilson(clicks, impressions):
    if impressions == 0:
        return 0
    else:
        return confidence(clicks, impressions)
def mplot(x,y,stdv):
    width =0.4
    plt.semilogx(x,y,'b--.')
    plt.errorbar(x, y, yerr=stdv)

    plt.ylabel('CTR')
    plt.xlabel('impressions')
    plt.title('CTR and confidence thereof vs impressions')
#    plt.xticks(ind+width/2., ('G1', 'G2', 'G3', 'G4', 'G5') )
#    plt.yticks(np.arange(0,81,10))
#    plt.legend( (p1[0], p2[0]), ('Men', 'Women') )

    plt.show()

if __name__ == '__main__':
    ctr = 1.0/10.0
    impressions = 1000
    x=[]
    y=[]
    stdvu=[]
    stdvd=[]
    answers = []
    i=1

    for j in range(0,20):
        print(ctr*i,i)
        res = wilson(ctr*i,i)
        x.append(i)
        y.append(ctr)
        stdvu.append(ctr-res[0])
        stdvd.append(res[1])
        print res
        answers.append(res[0])
        answers.append(res[1])
        i=i*2
    print(x,y,[stdvu,stdvd])
    mplot(x,y,[stdvu,stdvd])
    impressions = 100
    print(ctr*impressions,impressions)
    print wilson(ctr*impressions,impressions)
    impressions = 1000
    print(ctr*impressions,impressions)
    print wilson(ctr*impressions,impressions)
