import matplotlib.pyplot as plt
import numpy as np

target_object = "pen"

pose = np.load("./data/{}.npy".format(target_object))
data_num = pose.shape[0]
trans = pose[:, :3]
orientation = pose[:, 3:]

fig = plt.figure(figsize=(8.0, 6.0))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(np.arange(data_num), trans[:, 0], label="trans x")
ax1.plot(np.arange(data_num), trans[:, 1], label="trans y")
ax1.plot(np.arange(data_num), trans[:, 2], label="trans z")
ax1.set_ylabel("translation[m]")

ax2.plot(np.arange(data_num), orientation[:, 0], label="orientation x")
ax2.plot(np.arange(data_num), orientation[:, 1], label="orientation y")
ax2.plot(np.arange(data_num), orientation[:, 2], label="orientation z")
ax2.set_xlabel("t")
ax2.set_ylabel("euler angle")

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
plt.title("{} data".format(target_object), y=2.3)
plt.savefig('./data/{}.png'.format(target_object))