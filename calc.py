import torch 
import numpy as np

testauc = np.array([0.7278,0.7271,0.7282,0.7283])
testacc = np.array([0.7503,0.7507,0.7497,0.7506,0.7502])
testwindowauc = np.array([0.7279,0.7273,0.7288,0.7283,0.7284])
testwindowacc = np.array([0.7503,0.7505,0.7496,0.7505,0.7502])

print(np.average(testauc),'±',np.std(testauc))
print(np.average(testacc),'±',np.std(testacc))
print(np.average(testwindowauc),'±',np.std(testwindowauc))
print(np.average(testwindowacc),'±',np.std(testwindowacc))