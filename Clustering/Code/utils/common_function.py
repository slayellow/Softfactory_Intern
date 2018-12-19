import os
import time
import pandas as pd
import csv
import shutil

class CommonFunc():
    """Common Function Class

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
        pass

    def search_file(self, filepath, fileformat):
        """Draw Waveform Graph Class

        Parameters
        ----------
        filepath: data file path
        fileformat: data file format

        Returns
        -------
        file_list: file name saved in list
        """
        file_list = []
        for (path, dir, files) in os.walk(filepath):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.'+fileformat:
                    file = path + '/' + filename
                    file_list.append(file)
        return file_list

    def open_dat(self, filepath):
        """open .dat file and save data

        Parameters
        ----------
        filepath: data file path

        Returns
        data: data saved in list
        -------
        file_list: file name saved in list
        """
        print('Open the File : '+ filepath)
        file = open(filepath)
        reader = csv.reader(file, delimiter='\t')
        data = []
        for line in reader:
            tmp = line[0].split(',')
            int_data = []
            tmp = tmp[0:-1]
            for temp in tmp:
                int_data.append(float(temp))
            data.append(int_data)
        file.close()
        return data

    def open_csv(self, filepath):
        """open .csv file and save data

        Parameters
        ----------
        filepath: data file path

        Returns
        -------
        data_list: data saved in list
        """
        data_list = []
        f = open(filepath, 'r')
        rr = csv.reader(f)

        for read in rr:
            temp = [float(r) for r in read]
            if temp[-1] == 0 and temp[-2] == 0 and temp[-3] == 0 and temp[-4] == 0:
                continue
            else:
                data_list.append(temp)
        f.close()
        return data_list

    def make_dir(self, folderpath):
        """Make Folder

        Parameters
        ----------
        folderpath: folder path

        Returns
        -------
        """
        if not os.path.isdir(folderpath):
            os.mkdir(folderpath)

    def column(self, matrix, i):
        """Return all data in one column

        Parameters
        ----------
        matrix: data in list
        i: column index

        Returns
        -------
        data in list: [row[i] for row in matrix]
        file_list: file name saved in list
        """
        return [row[i] for row in matrix]