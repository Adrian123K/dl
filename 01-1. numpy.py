import numpy as np
a=[1,2,3,4]
a=np.array(a).reshape((2,2))
print(a)

print(a+5)

# 점심
import numpy as np
a = np.array([1,2,3])
b = np.array([2,4,6])
print('1', a+b)
print('2', a-b)
print('3', a*b)
print(a/b)

# 5
import numpy as np
a = np.array([2,4,8])
w = np.array([4,3,2])
k = a.dot(w)
k2 = np.sum(a*w)
print(k, k2)

# 7
import numpy as np
a = np.array([1,2])
w1 = np.array([[1,3,5],[2,4,6]])
w2 = np.array([[3,4],[5,6],[7,8]])
k = a.dot(w1)
m = k.dot(w2)
m

# 예제
a = [[51,55],[14,19],[0,4]]
for i in range(len(a)):
    for j in range(len(a[i])):
        if a[i][j]>=15:
            print(a[i][j])

# plt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20,10)
plt.rcParams.update({'font.size':22})

x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,9,8,3,2,4,3,4])

plt.scatter(x, y, color='red', s=80)
plt.show()

