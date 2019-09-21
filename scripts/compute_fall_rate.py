import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

v = pd.read_csv('fall-rate-sprint-027-2.txt', engine = 'python').values

print('Total Episodes: ' +  str(v.shape[0]))

print('Total Falls: ' + str(v[v > 0].shape[0]))

print('Fall Rate: ' + str(v.sum()/v.shape[0]))
