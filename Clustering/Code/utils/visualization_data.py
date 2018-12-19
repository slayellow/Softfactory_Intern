import matplotlib.pyplot as plt
import csv, os
import shutil
from PIL import Image
import Code.utils.common_function as cf

class DataGraph():
    """Draw Waveform Graph Class

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
        self.filepath = path
        self.filename = ''
        self.savelist = ['VOLTAGE', 'CURRENT']
        self.filelist = ['SagSwell파형', '고장파형', '특이파형', '중첩파형']
        self.foldername = ['ALL','CURRENT','VOLTAGE']
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
            file_split = file.split('/')
            data = self.cf.open_dat(file)
            for sl in self.savelist:
                self.draw_part_plot(data, file_split[-1], sl)
                #self.draw_plot(data, file_split[-1], sl)

    def draw_part_plot(self, data, filename, savetype):
        """Draw Waveform Data Graph

        Parameters
        ----------
        path: Waveform Data of List Type
        filename: Filename of List Type
        savetype: CURRENT or VOLTAGE

        Returns
        -------
        """
        """
       Flicker: 위치 상관X -- 256개
       Interruption: 1132:1388 -- 256개
       Sag: 1120:1246 / 1280:1408 -- 254개
       Swell: 1122:1240 / 1303:1440 -- 255개
       Notching: 1270:1334 * 4 -- 256개
       주파수변동: 640:896 -- 256개
       전압불평형: 1270:1526 -- 256개 
       """
        data_temp = self.cf.column(data, 7) # A상 - 6, B상 - 7, C상 - 8
        data1 = data_temp[1120:1246]
        data2 = data_temp[1280:1408]
        notching = data_temp[1270:1334]
        data3 = data_temp[640:896]

        folderpath = '../../' + savetype + '/'
        self.cf.make_dir(folderpath)
        folderpath = folderpath + filename
        print(folderpath)
        self.cf.make_dir(folderpath)
        plt.clf()
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(1,1,1)
        if '전압불평형' in filename:
            data_column = 2
            if savetype is 'VOLTAGE':
                data_column = 6
            plt.plot(range(256), self.cf.column(data, data_column)[1270:1526], color='b', label='Voltage_Unbalance_A')
            plt.plot(range(256), self.cf.column(data, data_column+1)[1270:1526], color='g', label='Voltage_Unbalance_B')
            plt.plot(range(256), self.cf.column(data, data_column+2)[1270:1526], color='r', label='Voltage_Unbalance_C')
        elif '중첩파형' in filename:
            data_column=2
            if savetype is 'VOLTAGE':
                data_column=5
            plt.plot(range(128),
                     self.cf.column(data, data_column)[:],
                     color='b', label=savetype + '_A')
            plt.plot(range(128),
                     self.cf.column(data, data_column + 1)[:],
                     color='g', label=savetype + '_B')
            plt.plot(range(128),
                     self.cf.column(data, data_column + 2)[:],
                     color='r', label=savetype + '_C')
        else:
            data_column = 2
            if savetype == 'VOLTAGE':
                data_column=6
            plt.plot(range(128), data3, color='b', label='Sag')
#            plt.plot(range(640), self.cf.column(data, data_column)[640:1280], color='b', label='dfa')
#            plt.plot(range(256), self.cf.column(data, data_column+1)[1132:1388], color='b', label='Flicker')
        plt.xlim(0,130)
        xtick = [i*32 for i in range(1, 5)]
       # ytick = [i*100000 for i in range(-8, 9)]
        plt.xticks(xtick)
      #  plt.yticks(ytick)
        ax.set_xlabel('128sample/cycle')
        ax.set_ylabel('Ampere(A)')
        plt.savefig(folderpath + '/' + filename + '_ABC.png')
        plt.close()
        self.convert_png_jpg(folderpath + '/' + filename + '_ABC.png')
        print(self.filename + "Finish to generate plot!")

    def draw_plot(self, data, filename, savetype):
        """Draw Waveform Data Graph

        Parameters
        ----------
        path: Waveform Data of List Type
        filename: Filename of List Type
        savetype: CURRENT or VOLTAGE

        Returns
        -------
        """
        folderpath = '../../' + savetype + '/'
        self.cf.make_dir(folderpath)
        folderpath = folderpath + filename
        print(folderpath)
        self.cf.make_dir(folderpath)
        plt.clf()
        plt.figure(figsize=(200, 40))
        if '중첩파형' in filename or '특이파형' in filename:
            data_column = 2
            if savetype is 'VOLTAGE':
                data_column = 5
            if '특이파형' in filename:
                length = 1000
                size = 40
            else:
                length = 64
                size = 4
            for i in range(int(len(data) / length)):
                plt.subplot(int(len(data) / length), 1, (i + 1))
                plt.plot(self.cf.column(data, 0)[
                         int(i * len(data) / int((len(data) / length))):int((i+1) * len(data) / int((len(data) / length)))],
                         self.cf.column(data, data_column)[
                         int(i * len(data) / int((len(data) / length))):int((i + 1) * len(data) / int((len(data) / length)))],
                         color='b', label=savetype+'_A')
                plt.plot(self.cf.column(data, 0)[
                         int(i * len(data) / int((len(data) / length))):int((i+1) * len(data) / int((len(data) / length)))],
                         self.cf.column(data, data_column+1)[
                         int(i * len(data) / int((len(data) / length))):int((i + 1) * len(data) / int((len(data) / length)))],
                         color='g', label=savetype+'_B')
                plt.plot(self.cf.column(data, 0)[
                         int(i * len(data) / int((len(data) / length))):int((i+1) * len(data) / int((len(data) / length)))],
                         self.cf.column(data, data_column+2)[
                         int(i * len(data) / int((len(data) / length))):int((i + 1) * len(data) / int((len(data) / length)))],
                         color='r', label=savetype+'_C')
                plt.legend(prop={'size': int(len(data) / size)})
        else:
            data_column = 2
            if savetype == 'VOLTAGE':
                data_column=6
            for i in range(int(len(data) / 1000)):
                plt.subplot(int(len(data) / 1000), 1, (i + 1))
                plt.plot(self.cf.column(data, 0)[
                         int(i * len(data) / int((len(data) / 1000))):int((i + 1) * len(data) / int((len(data) / 1000)))],
                         self.cf.column(data, data_column)[
                         int(i * len(data) / int((len(data) / 1000))):int((i + 1) * len(data) / int((len(data) / 1000)))],
                         color='b', label=savetype+'_A')
                plt.plot(self.cf.column(data, 0)[
                         int(i * len(data) / int((len(data) / 1000))):int((i + 1) * len(data) / int((len(data) / 1000)))],
                         self.cf.column(data, data_column+1)[
                         int(i * len(data) / int((len(data) / 1000))):int((i + 1) * len(data) / int((len(data) / 1000)))],
                         color='g', label=savetype+'_B')
                plt.plot(self.cf.column(data, 0)[
                         int(i * len(data) / int((len(data) / 1000))):int((i + 1) * len(data) / int((len(data) / 1000)))],
                         self.cf.column(data, data_column+2)[
                         int(i * len(data) / int((len(data) / 1000))):int((i + 1) * len(data) / int((len(data) / 1000)))],
                         color='r', label=savetype+'_C')
                if len(data) > 3000:
                    plt.legend(prop={'size': 10})
                else:
                    plt.legend(prop={'size': 40})

        plt.savefig(folderpath + '/' + filename + '_ABC.png')
        plt.close()
        self.convert_png_jpg(folderpath + '/' + filename + '_ABC.png')
        print(self.filename + "Finish to generate plot!")

    def convert_png_jpg(self, filename):
        """Generate .jpg file and Remove .png file

        Parameters
        ----------
        filename: .png filename

        Returns
        -------
        """
        im = Image.open(filename)
        rgb_im = im.convert('RGB')
        rgb_im.save(filename+'.jpg')
        if os.path.isfile(filename):
            os.remove(filename)
            print(filename + ' Remove!')

    def filecopy(self, foldername):
        """Save all image files to a new folder

        Parameters
        ----------
        foldername: folder name to save

        Returns
        -------
        """
        for  sl in self.savelist:
            for fl in self.filelist:
                self.cf.make_dir('./'+foldername+'_' + sl + '_' + fl +'/')
                for (path, dir, files) in os.walk('./'+sl):
                    for filename in files:
                        ext = os.path.splitext(filename)[-1]
                        if ext == '.jpg':
                            if fl in filename:
                                shutil.copy(path + '/' + filename, './' + foldername + '_' + sl + '_' + fl +'/')


path = 'D://Data/Collect_Data/Collect_중첩파형'
ssw = DataGraph(path)
ssw.filecopy('Plot')