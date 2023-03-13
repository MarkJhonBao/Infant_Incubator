import numpy as np
name_list = ['头部', '左手', '左腿', '右手', '右腿']
prob_result = np.array([[1.5974808e-01,3.3728755e-04,7.9550268e-03,9.7646207e-01,9.9998558e-01]])
prob_result = prob_result.reshape(prob_result.shape[0]*prob_result.shape[1], )
probs = []
# prob_result =[0,0,0,0,0]
res = str
name_list = ['头部', '左手', '左腿', '右手', '右腿']

for i in range(len(prob_result)):
    if prob_result[i] > 0.5:
        prob_result[i] = prob_result[i]
        probs.append(name_list[i] + '(' + str("%.2f" % (prob_result[i] * 100)) + '%)')
if len(probs) :
    res = ('+'.join(probs))
else:
    res = "正常"




print(res)