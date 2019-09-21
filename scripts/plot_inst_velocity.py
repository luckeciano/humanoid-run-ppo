import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

v = pd.read_csv('run_statistics.csv', engine = 'python')

epidodes = sorted(v['episode'].unique())

P = 1
STEP_TIME = 0.02

for e in epidodes:
	steps = v[v['episode'] == e][['x']]
	inst_velocities = steps.diff(periods=P).fillna(0)/(P * STEP_TIME)
	inst_velocities = inst_velocities[inst_velocities > 0.0]
	inst_velocities = inst_velocities[inst_velocities < 5.0]
	plt.plot(inst_velocities)
	break


# plt.legend('trajectories')
plt.grid()
plt.title('Instantaneous Forward Speed')
plt.xlabel('x')
plt.ylabel('y')
# plt.yticks(np.arange(1.350, 3.86, .250))
plt.show()
