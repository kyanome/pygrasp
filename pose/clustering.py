import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import glob 
import matplotlib.pyplot as plt

def get_data():
    files = glob.glob('./data_collect/data/*.npy')
    data_num = 20
    pose_list = []
    for f in files:
        pose = np.load(f)[:data_num, :]
        pose_list.append(pose)
    pose_list = np.array(pose_list).reshape((len(files)*data_num, 6))

    return pose_list

def main():
    pose = get_data()
    pred = KMeans(n_clusters=8).fit_predict(pose)

    data_num = pose[pred==2].shape[0]
    time_data = np.arange(data_num)
    #plt.plot(time_data, pose[pred==2][:, 4])
    #plt.show()

    data_num = pose[pred==3].shape[0]
    time_data = np.arange(data_num)
    #plt.plot(time_data, pose[pred==3][:, 4])
    #plt.show()

    # ３次元特徴を2次元にする
    pca = PCA(n_components=2)
    pca.fit(pose)
    pca_data = pca.fit_transform(pose)


    plt.figure()
    cmap = plt.get_cmap("tab10")
    for i in range(pca_data.shape[0]):
        print(i)
        plt.scatter(pca_data[i,0], pca_data[i,1], c=cmap(pred[i]))
    plt.show()

if __name__ == "__main__":
    main()