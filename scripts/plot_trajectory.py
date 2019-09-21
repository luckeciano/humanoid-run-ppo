import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

v = pd.read_csv('run-stats-sprint-027-2.csv', engine = 'python')

episodes = sorted(v['episode'].unique())
episodes = filter(lambda x : x <= 100, episodes)

deviation = []

for e in episodes:
	steps = v[v['episode'] == e][['x', 'y']]
	first, last = steps.iloc[0], steps.iloc[-1]
	deviation.append(np.degrees(np.arctan( math.fabs(last['y'] - first['y'])/math.fabs(last['x'] - first['x']))))
	plt.plot(steps['x'], steps['y'])

print(np.mean(deviation), np.std(deviation))

# plt.legend('trajectories')
plt.grid()
plt.title('Trajectories - Sprint')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.yticks(np.arange(-10.0, 10.01, 2.5))
plt.savefig('sprint-traj-0.27-2.eps', format='eps')
plt.show()
