import pandas as pd
import matplotlib.pyplot as plt




env_name = "buffer1e5_rewardOrigin_gamma99_2"
data = pd.read_csv("./preTrained/{}/log.txt".format(env_name),delimiter = ",", engine='python')
plt.figure(figsize = (20,10))
plt.plot(data.iloc[:,1],label='buffer 1e5')
plt.legend(loc='best')
env_name = "buffer2e5_rewardOrigin_gamma99_2"
data = pd.read_csv("./preTrained/{}/log.txt".format(env_name),delimiter = ",", engine='python')
#plt.figure(figsize = (20,10))
plt.plot(data.iloc[:,1],label='buffer 2e5')
plt.legend(loc='best')
env_name = "buffer3e5_rewardOrigin_gamma99_2"
data = pd.read_csv("./preTrained/{}/log.txt".format(env_name),delimiter = ",", engine='python')
#data = data.iloc[1200:1534,:]
#plt.figure(figsize = (20,10))
plt.plot(data.iloc[:,1],label='buffer 3e5')
plt.legend(loc='best')
print("save")
plt.savefig("./preTrained/{}/reward.png".format(env_name))


#4240