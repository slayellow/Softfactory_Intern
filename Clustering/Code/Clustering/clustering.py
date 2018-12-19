from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing as preprocessing
from sklearn.cluster import DBSCAN
import Code.utils.common_function as cf

class Clustering():
    """Clustering Class

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self):
        """Class Constructor

        Parameters
        ----------

        Returns
        -------
        """
        self.cf = cf.CommonFunc()
        self.data_list = []
        self.colors = ['black', 'blue', 'purple','red', 'yellow']
        self.labels = ['Original','Fault','Unique','Swell','Sag']
#        self.labels = ['Cluster_1','Cluster_2','Cluster_3','Cluster_4','Cluster_5']
        self.folderpath = 'CSV'
        self.waveform = ['SagSwell파형', '고장파형', '특이파형']
        self.phasefile = ['VOLTAGEA_Result.csv','VOLTAGEB_Result.csv','VOLTAGEC_Result.csv']
        self.process()

    def process(self):
        """Process Flow

        Parameters
        ----------

        Returns
        -------
        """
        temp = []
        for wave in self.waveform:
            data = self.cf.open_csv('./'+self.folderpath+'/'+wave +'/'+ self.phasefile[0])
            for dat in data:
                temp.append(dat)
            print(wave + ' Data Length : '+str(len(data)))
            self.data_list.extend(data)
        #self.data_list = self.L2_Normalization(self.data_list)
        sort_list = self.sort(self.data_list)
        data = self.PCA(sort_list, 2)
        data = self.Y_Normalization(data)
        for i in range(2,6):
            print('Clustering Start : ',str(i))
            y_predict = self.KMeans(data, i)      # K-Means Clustering
            # self.draw_clustering_plot(data,i, y_predict, 'Clustering_'+str(i))
        self.draw_cluster_data_to_plot(temp, y_predict)

    def sort(self, data_list):
        """Sort data in one row

        Parameters
        ----------
        data_list: data in list

        Returns
        -------
        sort_list: sorted data in list
        """
        sort_list = []
        for data in data_list:
            data.sort()
            sort_list.append(data)
        return sort_list

    def draw_cluster_data_to_plot(self, data_list, y_predict):
        """Draw data graph showing clustered results.

        Parameters
        ----------
        data_list: sorted data in list
        y_predict: list of results predicted through clustering

        Returns
        -------
        """
        plt.figure()
        count = 0
        for i, data in enumerate(data_list):
            if count == 20:
                print('Count :' + str(i))
                plt.legend()
                plt.savefig('./Graph/Cluster_Sort_Graph/Image_' + str(i) + 'img.png')
                count = 0
                plt.clf()
            plt.plot(range(32), data[:], label=str(i)+' Prediction : ' + str(y_predict[i]))
            count += 1
        plt.legend()
        plt.savefig('./Graph/Cluster_Sort_Graph/Image_' + str(i) + 'img.png')
        plt.clf()
        plt.close()

    def PCA(self, data, n_component):
        """Do a Principal Component Analysis on data and Save in the list

        Parameters
        ----------
        data: data in list
        n_component: The number of PCA

        Returns
        -------
        X_pca: result list through PCA
        """
        estimator = PCA(n_components=n_component)
        X_pca = estimator.fit_transform(data)
        print("주성분 분석 결과 설명력 : " + str(estimator.explained_variance_ratio_))
        return X_pca

    def Y_Normalization(self, data):
        """ Normalize to the Y axis

        Parameters
        ----------
        data: data in list

        Returns
        -------
        data: result data normalized to the Y axis
        """
        data = np.array(data)
        data_Y = data[:,1]
        data_list = np.array(data_Y)
        data_list = np.reshape(data_list, [1, -1])
        data_Y = preprocessing.normalize(data_list, norm='l2')
        data_Y = np.reshape(data_Y, [-1,1])
        for i in range(len(data_Y)):
            data[i,1] = data_Y[i]
        return data

    def KMeans(self, data, n_cluster=2):
        """Do K-Means Clustering

        Parameters
        ----------
        data: data in list
        n_cluster: The number of cluster

        Returns
        -------
        y_predict: The predicted value of the list type
        """
        model = KMeans(n_clusters=n_cluster, algorithm='auto', random_state=20, max_iter=10).fit(data)
        y_predict = model.predict(data)
        return y_predict

    def draw_clustering_plot(self, data, n_cluster, y_predict, savename):
        """Draw a Distribution of Clustering Results

        Parameters
        ----------
        data: data in list
        n_cluster: The number of cluster
        y_predict: The predicted value of the list type
        savename: The name of the image file to save

        Returns
        -------
        """
        print('Data Visualization!')
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(111)
        for i in range(n_cluster):
            px = data[:, 0][y_predict == i]
            py = data[:, 1][y_predict == i]                     # 2차원 차원축소시 사용
            #py = [1 for _ in range(px.shape[0])]               # 1차원 차원축소시 사용
            ax.scatter(px[:], py[:], c=self.colors[i], label=self.labels[i])
        plt.legend()
        print('Draw!!')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        # ax.set_title('Silhouette Score: {:.3f}'.format(silhouette_score(data, y_predict)))
        plt.savefig(savename+'.png')
        plt.close()
        print('Finish Data Visualization!!')

    def Min_Max_Normalization(self, data):
        """Min-Max Normalize to the data

        Parameters
        ----------
        data: data in list

        Returns
        -------
        data_result_list: result data normalized to the data
        """
        data_inverse_list = [list(i) for i in [*zip(*data)]]
        data_inverse_list = MinMaxScaler().fit_transform(data_inverse_list)
        data_result_list = [list(i) for i in [*zip(*data_inverse_list)]]
        return data_result_list

    def L2_Normalization(self, data):
        """L2 Normalize to the data

        Parameters
        ----------
        data: data in list

        Returns
        -------
        data_result_list: result data normalized to the data
        """
        data_list = np.array(data)
        row, column = data_list.shape
        data_list = np.reshape(data_list, [1, -1])
        data_list = preprocessing.normalize(data_list, norm='l2')
        return np.reshape(data_list,[row,column])


Clustering()