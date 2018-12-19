import csv
import os
import Code.utils.common_function as cf

class Preprocessing():
    """Preprocessing Class

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self, path):
        """Class Constructor

        Parameters
        ----------
        path: data file path

        Returns
        -------
        """
        self.cf = cf.CommonFunc()
        self.savefolder = 'CSV'
        self.savetype = ['CURRENT','VOLTAGE']
        self.phase = ['A', 'B', 'C']
        self.VOLTAGE_A = []
        self.VOLTAGE_B = []
        self.VOLTAGE_C = []
        self.process(path)


    def process(self, path):
        """Process Flow

        Parameters
        ----------
        path: data file path

        Returns
        -------
        """
        file_list = self.cf.search_file(path, 'dat')
        for file in file_list:
            data = self.cf.open_dat(file)
            if '중첩' in file or '특이' in file:
                self.convert_row(self.cf.column(data, 5), self.VOLTAGE_A)
                self.convert_row(self.cf.column(data, 6), self.VOLTAGE_B)
                self.convert_row(self.cf.column(data, 7), self.VOLTAGE_C)
            else:
                self.convert_row(self.cf.column(data, 6), self.VOLTAGE_A)
                self.convert_row(self.cf.column(data, 7), self.VOLTAGE_B)
                self.convert_row(self.cf.column(data, 8), self.VOLTAGE_C)

        folderpath = 'D://Code/2018_11_19_Clustering/Clustering/'+self.savefolder + '/SagSwell파형_이상주기'         # 저장시킬 위치 나중에 수정해야함
        self.cf.make_dir(folderpath)
        self.save_csv(self.VOLTAGE_A, folderpath+'/'+self.savetype[1]+self.phase[0])
        self.save_csv(self.VOLTAGE_B, folderpath+'/'+self.savetype[1]+self.phase[1])
        self.save_csv(self.VOLTAGE_C, folderpath+'/'+self.savetype[1]+self.phase[2])

    def convert_row(self, data, savelist):
        """Save 32 samples in one row

        Parameters
        ----------
        data: Waveform data of one column
        savelist: The name of the list variable to be saved

        Returns
        -------
        """
        count = 0
        tmp = []
        for i, dat in enumerate(data):
            tmp.append(dat)
            count += 1
            if count == 32:
                savelist.append(tmp)
                tmp = []
                count = 0

    def save_csv(self, data_list, filepath):
        """Save data to .csv file

        Parameters
        ----------
        data_list: List type data changed to a row
        filepath: .csv file name

        Returns
        -------
        """
        tmp = []
        count = 0
        for read in data_list:
            count += 1
            if float(read[0]) == 0 and float(read[1]) == 0 and float(read[2]) == 0 and float(read[3]) == 0 and float(
                    read[4]) == 0:
                continue
            elif float(read[-1]) == 0 and float(read[-2]) == 0 and float(read[-3]) == 0 and float(read[-4]) == 0:
                continue
            else:
                tmp.append(read)
        f = open(filepath + "_Result.csv", 'w', encoding='utf-8', newline='')  # 0인 부분을 제외하고 새로운 .csv파일에 저장
        wr = csv.writer(f)
        for temp in tmp:
            wr.writerow(temp)
        f.close()


pp = Preprocessing('D://Data/Collect_Data/Collect_SagSwell파형_이상주기')        # 파라메타값을 다르게 설정
