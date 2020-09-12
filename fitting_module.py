""" フィッティングモジュール """
import os
import csv
import datetime
import pandas as pd
from sklearn.metrics import mean_absolute_error #機械学習ライブラリ 平均絶対誤差
#from sklearn.metrics import mean_squared_error #機械学習ライブラリ 平均二乗誤差
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
np.set_printoptions(precision=1) #numpyの表示桁数指定

class Fit:
    """ フィッティング """

    def __init__(self, filename, guess, background):
        self.filename = filename
        self.df = pd.read_csv(filename+'.csv')
        self.x, self.y = map(lambda i: self.df[self.df.columns[i]], list(range(len(self.df.columns))))

        #初期値リストの結合
        self.guess_all = []
        for i in guess:
            self.guess_all.extend(i)
        self.guess_all.append(background)

        #初期値のパラメーターとMAE
        _, mae = self.superposition(self.x, self.y, *self.guess_all)  #ガウス関数の重ね合わせ, 平均絶対誤差
        print("初期値：", self.guess_all, ",    mean：", mae)

        #フィッティングを実行
        self.popt, self.pcov = curve_fit(self.func, self.x, self.y, p0=self.guess_all, maxfev=10000) #パラメーター, 共分散

        #フィッティングしたのパラメーターとMAE
        self.y_total, self.mae = self.superposition(self.x, self.y, *self.popt)   #ガウス関数の重ね合わせ, 平均絶対誤差
        print("Result：", self.popt, ",   mean：", self.mae)

        self.y_list = self.func_list(self.x, *self.popt)

        self.out_path = 'output'
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)

    def func_list(self, x, *params):
        """ 各ガウス関数を定義 """

        num_func = int(len(params)/3)   #paramsの長さでフィッティングする関数の数を判別。

        #ガウス関数にそれぞれのパラメータを挿入してy_listに追加。
        y_list = []
        for i in range(num_func):
            yi = np.zeros_like(x)
            param_range = list(range(3*i, 3*(i+1), 1))
            amp = params[int(param_range[0])]
            ctr = params[int(param_range[1])]
            wid = params[int(param_range[2])]
            yi = yi + amp * np.exp(-((x - ctr)/wid)**2) + params[-1]
            y_list.append(yi)
        return y_list

    def func(self, x, *params):
        """ 全ガウス関数の重ね合わせとバックグラウンドの追加 """

        #ガウス関数にそれぞれのパラメータを挿入してy_listに追加。
        y_list = self.func_list(x, *params)

        #y_listに入っているすべてのガウス関数を重ね合わせる。
        y_sum = np.zeros_like(x)
        for i in y_list:
            y_sum = y_sum + i

        #最後にバックグラウンドを追加。
        y_sum = y_sum + params[-1]
        return y_sum

    def superposition(self, x, y, *params):
        """ RAWデータとフィッティング関数との平均絶対誤差 """
        y_sum = self.func(x, *params)
        return y_sum, mean_absolute_error(y, y_sum)

    def to_csv(self, flag):
        """ プロットを保存 """
        if flag:
            for i, n in enumerate(self.y_list):
                self.df['Gauss'+str(i+1)] = n
            self.df['Gauss total'] = self.y_total
            self.df.to_csv(self.out_path+'/'+self.filename+'_fit.csv', index=False)

    def save_parameter(self, flag):
        """ パラメーターを保存 """
        if flag:
            summary = self.out_path+'/summary.csv'
            col = [datetime.datetime.now().strftime('%Y%m%d%H%M%S'), self.filename]
            col.extend(self.popt)
            res = [col]

            if os.path.isfile(summary):                     # 過去データの読み込み
                with open(summary) as file:
                    reader = csv.reader(file)
                    for row in reader:
                        res.append(row)

            with open(summary, "w", newline="") as file:    # ファイルに保存
                writer = csv.writer(file)
                for row in res:
                    writer.writerow(row)

class Plot:
    """ プロット """

    def __init__(self, res):
        self.filename = res.out_path+'/'+res.filename
        self.fig = self.show(res)

    def show(self, res):
        """ フィッティング用関数の定義 """
        #プロット図用意
        fig = plt.figure()

        #RAWデータのプロット
        plt.scatter(res.x, res.y, s=20)

        #重ね合わせたフィッティングのプロット
        plt.plot(res.x, res.func(res.x, *res.popt), ls='-', c='black', lw=1)

        #各フィッティングのプロット
        baseline = np.zeros_like(res.x) + res.popt[-1]
        for i, yi in enumerate(res.y_list):
            plt.fill_between(res.x, yi, baseline, facecolor=cm.get_cmap('rainbow')(i/len(res.y_list)), alpha=0.6)
        return fig

    def save(self, flag):
        """ プロット図の保存 """
        if flag:
            self.fig.patch.set_alpha(0)
            self.fig.savefig(self.filename)
