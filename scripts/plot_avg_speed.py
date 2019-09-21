import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

v = pd.read_csv('run_statistics.csv', engine = 'python')

episodes = sorted(v['episode'].unique())
episodes = filter(lambda x : x <= 1000, episodes)
STEP_TIME = 0.02

def avg_velocity_from_k(episodes, k = 0.0):
	avg_vel = []

	for e in episodes:
		steps = v[v['episode'] == e][['x', 'y']]
		steps = steps[steps['x'] >= k]
		if steps.shape[0] < 200:
			continue
		last, first = steps.iloc[-1], steps.iloc[0]
		avg_velocity = (last['x'] - first['x'])/((last.name - first.name) * STEP_TIME)
		avg_vel.append(avg_velocity)

	bs_mean_step = bs.bootstrap(np.array(avg_vel), stat_func=bs_stats.mean, alpha=0.05)
	bs_std_step = bs.bootstrap(np.array(avg_vel), stat_func=bs_stats.std, alpha=0.05)

	return bs_mean_step.value, bs_std_step.value, bs_mean_step.upper_bound, bs_mean_step.lower_bound, np.max(avg_vel), np.min(avg_vel)

mean_vel, std_vel, max_vel, min_vel, ub_vel, lb_vel = [], [], [], [], [], []

for k in np.arange(-14.0, -11.0, 0.5):
	a, b, c, d, e, f = avg_velocity_from_k(episodes, k)
	mean_vel.append(a)
	std_vel.append(b)
	ub_vel.append(c)
	lb_vel.append(d)
	max_vel.append(e)
	min_vel.append(f)


start_line = np.arange(0.0, 3.0, 0.5)
# plt.legend('trajectories')
print(mean_vel, max_vel, std_vel)
plt.plot(start_line, mean_vel)
plt.plot(start_line, max_vel, c='g', ls='dashed')
#plt.plot(start_line, min_vel, c='g', ls='dashed')
plt.fill_between(start_line, lb_vel, ub_vel,  alpha = 0.4)
plt.grid()
plt.title('Forward Speed Evaluation')
plt.xlabel('Starting line (m)')
plt.ylabel('Average Forward Speed (m/s)')
plt.legend(['Average', 'Max', '95% CI'])
# plt.yticks(np.arange(1.350, 3.86, .250))
plt.savefig('sprint-speed-eval-0.33-1.pdf', format='pdf')
plt.show()
