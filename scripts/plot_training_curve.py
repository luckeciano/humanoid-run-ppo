import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

v = pd.read_csv('paper-experiment-2', engine = 'python').values

x = range(v.shape[0])


plt.scatter(x, v, s = [0.2])
plt.axhline(y = 2.6, ls=':', c='r')
plt.legend(['state-of-the-art', 'episodes'])
plt.grid()
plt.title('Reward Curve - Sprint')
plt.xlabel('Episode')
plt.ylabel('Forward Speed (m/s)')
plt.yticks(np.arange(1.350, 3.86, .250))
#plt.show()
plt.savefig('sprint-train-0.27-2.eps', format='eps')
