import scipy.io as spio

case2 = spio.loadmat('case2_01.mat')
case4 = spio.loadmat('case3_01.mat')

estimate = case4['estimates']
prob = case4['estimate_value']

for i in range(20):
    print "I-%d & %d / %f & %d / %f & %d / %f \\\\" % (i+1,estimate[i][0],prob[i][0],estimate[i][1],prob[i][1],estimate[i][2],prob[i][2])
    print "\hline"
