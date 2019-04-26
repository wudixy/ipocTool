#!/usr/bin/python
#-*- coding: utf-8 -*

import pandas as pd
import numpy as np
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,DayLocator, MONDAY
import matplotlib.dates as mdates
#from mpl_finance import candlestick_ohlc
import mpl_finance as mpf
import datetime
import glob
import time
import argparse

modeldir = './model'
reportdir = './imgReport'

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def showplt(df,topic, show=False, Save=''):
    #将时间数据转换为matplotlib的时间格式
    model = df.sort_values(by='Date',ascending=True)
    #model['day'] = model['Date'].apply(lambda d: mdates.date2num(d.to_pydatetime()))
    model['day'] = model['Date'].apply(lambda x:mdates.date2num(x))
    data=model[['day','start','end','max','min','trend','usedays']]
    #tuples = [tuple(x) for x in model[['day','start','end','max','min']].values]
    #data_mat=data.as_matrix()
    data_mat=data.values
    #绘制K线图
    #fig,(ax1,ax2,ax3)=plt.subplots(3,sharex=True,figsize=(1200/72,480/72))
    fig,(ax1,ax2)=plt.subplots(2,sharex=True,figsize=(1200/72,480/72))
    mpf.candlestick_ochl(ax1,data_mat,colordown='#ff1717', colorup='#53c156',width=0.3,alpha=1)
    ax1.grid(True)
    plt.title(topic)
    ax1.xaxis_date()

    # 生成连续分布趋势数据
    pos, = np.where(np.diff(model.trend_flag))
    start,end = np.insert(pos+1,0,0),np.append(pos,len(model)-1)
    x = pd.DataFrame({'start':list(model.day.iloc[start]),
                      'end':list(model.day.iloc[end]),
                      'n':end-start+1,
                      'flag': list(model.trend_flag.iloc[start])
                      })
    rec_r = x[(x.n>=3) & (x.flag ==1)].reset_index()
    for ix,row in rec_r.iterrows():
        rec_x = row['start']
        rec_y = data[data.day==row['end']]['min'].values[0]
        rec_w = row['end']-row['start']
        rec_h = data[data.day==rec_x]['max'].values[0]-data[data.day==row['end']]['min'].values[0]
        rect = plt.Rectangle((rec_x,rec_y), rec_w, rec_h,
                             fill=False, edgecolor = 'y',linewidth=1)
        ax1.add_patch(rect)
    # 绘制预警柱状图
    data_i = data[data.trend < 1].values
    data_w = data[(data.trend >= 1) & (data.usedays>7)].values
    data_e = data[(data.trend >= 1) & (data.usedays<=7)].values
    #data_i = data[(data.trend < 1) & (data.usedays>7)].values
    #data_w = data[(data.trend >= 1) & (data.usedays>7)].values
    #data_e = data[(data.trend >= 1) & (data.usedays<=7)].values
    #plt.bar(data_mat[:,0],data_mat[:,5],width=0.5)
    ax2.bar(data_i[:,0],data_i[:,5],width=0.5,color='g')
    ax2.bar(data_w[:,0],data_w[:,5],width=0.5, color='y')
    ax2.bar(data_e[:,0],data_e[:,5],width=0.5, color='r')
    #plt.hlines(1,x.min(),x.max(),linestyles = "dashed",colors='r')
    ax2.set_ylabel('trend')
    ax2.grid(True)

    for d in data_e:
        #print d
        plt.annotate('%0.2fdays'%(d[6]), xy=(d[0],d[5]), xytext=(d[0],d[5]))
    """
    #with pd.ExcelWriter(r'e:\tmp.xlsx') as writer:  # doctest: +SKIP
    #    model.to_excel(writer, sheet_name='all')
    #    x.to_excel(writer, sheet_name='trend')
    ax3.bar(data_mat[:,0],data_mat[:,6],width=0.5)
    #plt.hlines(1,x.min(),x.max(),linestyles = "dashed",colors='r')
    ax3.set_ylabel('base score')
    ax3.grid(True)
    """
    if show:
        plt.show()
    if Save:
        plt.savefig(Save.replace(' ','_')+'.pdf', format='pdf')


def diskTrend2(d, histday='30d',sigma=2):
    # 读取数据
    startt = time.time()
    data_list = []
    #files = glob.glob(os.path.join(d,'*-jfstrend.csv'))
    files = glob.glob(os.path.join(d,'*[!old]-jfstrend.csv'))
    for f in files:
        df = pd.read_csv(f, index_col=None, header=0)
        host = os.path.basename(f).split('-')[0]
        df['host'] = host
        data_list.append(df)
    if data_list:
        df = pd.concat(data_list,axis=0, ignore_index=True)

    print 'read data need %0.2f' %(time.time()-startt)
    #计算基础Kpi

    # 每日用量
    startt = time.time()
    df['pusage'] = df['start'] - df['end']
    #每日最大用量
    df['musage'] = df['max'] - df['min']
    #方向判断
    df['direction'] = np.sign(df['pusage'])
    #剔除用量为负的数值
    df.loc[df[df['direction']==-1].index,['pusage','musage']] = 0
    df['Date'] = pd.to_datetime(df['Date'],format='%Y%m%d')
    df.sort_values(by=['host','JFS','Date'], inplace=True)
    df.drop_duplicates(['host','JFS','Date'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    print 'base calc %0.2f' %(time.time()-startt)

    print '----------------base data overview---------------------'
    print 'host: %d' % (df['host'].unique().shape[0])
    print 'jfs: %d' % (df.drop_duplicates(['host','JFS']).shape[0])
    print 'records:%d' % (df.shape[0])
    print 'min date:' + str(df['Date'].min())
    print 'max date:' + str(df['Date'].max())
    print df.head()
    #df = df[df.JFS=='DISK_D']

    print '----------------model data overview---------------------'
    startt = time.time()
    df.set_index('Date',inplace=True)
    m_agv = df.groupby(by=['host','JFS']).rolling(histday,closed='left', min_periods=7).mean()[['pusage','musage']].reset_index().rename(columns={'pusage':'pmean','musage':'mmean'})
    m_std = df.groupby(by=['host','JFS']).rolling(histday,closed='left', min_periods=7).std()[['pusage','musage']].reset_index().rename(columns={'pusage':'pstd','musage':'mstd'})
    #print m1.xs(('NOVPLODS01','DISK_D'),level=[0,1])
    model = pd.merge(m_agv,m_std,how='inner',on=['host','JFS','Date'])
    model = pd.merge(df.reset_index(),model,how='left',on=['host','JFS','Date'])
    print 'rolling data need %0.2f' %(time.time()-startt)

    startt = time.time()
    model['p_score'] = (model['pusage']-model['pmean'])/(sigma*model['pstd'])
    model['m_score'] = (model['musage']-model['mmean'])/(sigma*model['mstd'])
    print 'score data need %0.2f' %(time.time()-startt)

    startt = time.time()
    #model['score'] = model.apply(lambda x: max(x['p_score'],x['m_score']), axis=1)
    model['score'] = np.maximum(model.p_score,model.m_score)
    print 'max score data need %0.2f' %(time.time()-startt)

    startt = time.time()
    td = model.set_index('Date').groupby(by=['host','JFS'])['score'].rolling('2d').mean().reset_index().rename(columns={'score':"trend"})
    print 'trend data need %0.2f' %(time.time()-startt)

    startt = time.time()
    model = pd.merge(model,td,how='left',on=['host','JFS','Date'])
    model['usedays'] = model['end'] / model['musage']
    model.sort_values(by=['host','JFS','Date'],ascending=True,inplace=True)
    model['trend_flag'] = model['trend'].apply(lambda x : 1 if x and x >=1 else 0)
    print 'all data need %0.2f' %(time.time()-startt)
    print model.head()

    def trendDist(dt,host,jfs):
        # 生成连续分布趋势数据
        pos, = np.where(np.diff(dt.trend_flag))
        start,end = np.insert(pos+1,0,0),np.append(pos,len(dt)-1)
        x = pd.DataFrame({'start':list(dt.Date.iloc[start]),
                          'end':list(dt.Date.iloc[end]),
                          'n':end-start+1,
                          'flag': list(dt.trend_flag.iloc[start]),
                          'host':host,
                          'JFS':jfs
                          })
        return x

    startt = time.time()
    rglist = []
    for nm,gp in model.groupby(by=['host','JFS']):
        rglist.append(trendDist(gp,nm[0],nm[1]))
    trg = pd.concat(rglist,axis=0, ignore_index=True)
    trg.to_csv(os.path.join(modeldir, 'diskUseTrend.csv'))
    model.to_csv(os.path.join(modeldir, 'model.csv'))

    #主要异常信息显示

    #特殊用例显示
    #h =  model[(model.host=='NOVPWATT02') & (model.JFS=='DISK_D')]
    #h.to_csv(r'e:\tmp.csv')
    #showplt(h,'NOVPWATT02 DISK_D')

    # 寻找连续三天趋势改变的
    # 连续三天趋势改变
    upper3days = trg[(trg.flag == 1) & (trg.n >= 3)].sort_values(by='n', ascending=False)
    upper3days.to_csv(os.path.join(reportdir,'uppder3dasyTrend.csv'))
    for ix,row in upper3days.iterrows():
        hs = row['host']
        js = row['JFS']
        stdt = row['start'] - datetime.timedelta(days=30)
        enddt = row['end'] + datetime.timedelta(days=10)
        print hs,js,stdt,enddt
        h =  model[(model.host==hs) & (model.JFS==js) & (model.Date >= stdt) & (model.Date <= enddt)]
        #print h.head() 
        tp = hs + '-' + js + str(row['start'])[0:10] +'-'+ str(row['end'])[0:10] +'-rangeUpper3days'
        showplt(h,tp,show=False,Save=os.path.join(reportdir,tp))
        #raw_input()



"""
    trendRange = trg[trg.flag == 1].groupby(by=['host','JFS']).count().sort_values(by=['n'], ascending=False)
    print trendRange.head()
    print 'dist data need %0.2f' %(time.time()-startt)


    # 选择重点机器生成图表
    topn = 2

    # 连续趋势改变天数
    #print trendRange.head()
    for ix,row in trendRange.head(topn).iterrows():
        h =  model[(model.host==ix[0]) & (model.JFS==ix[1])]
        topic = ix[0] + '_' + ix[1] + ' trendRange sum top'
        showplt(h, topic , show=False, Save=os.path.join(reportdir,topic))


    #趋势sum总高
    select_tope = model.groupby(by=['host','JFS'])['trend'].sum().reset_index().sort_values(by='trend')
    for ix,row in select_tope.head(1).iterrows():
        h =  model[(model.host==row['host']) & (model.JFS==row['JFS'])]
        topic = row['host'] + '_' + row['JFS'] + ' Trend sum top'
        showplt(h, topic, show=False,Save=os.path.join(reportdir,topic) )

    for ix,row in select_tope.tail(topn).iterrows():
        h =  model[(model.host==row['host']) & (model.JFS==row['JFS'])]
        topic = row['host'] + '_' + row['JFS'] + 'Trend sum tail'
        showplt(h,topic, Save=os.path.join(reportdir,topic))

    # 剩余磁盘可用天数最少
    select_tope = model.groupby(by=['host','JFS'])['usedays'].sum().reset_index().sort_values(by='usedays')
    for ix,row in select_tope.head(1).iterrows():
        h =  model[(model.host==row['host']) & (model.JFS==row['JFS'])]
        topic=row['host'] + '_' + row['JFS'] + ' usedasy sum top'
        showplt(h,topic,Save=os.path.join(reportdir,topic))

    for ix,row in select_tope.tail(topn).iterrows():
        h =  model[(model.host==row['host']) & (model.JFS==row['JFS'])]
        topic = row['host'] + '_' + row['JFS'] + 'usedasy sum tail'
        showplt(h, topic, Save=os.path.join(reportdir,topic))
"""



#d = r'E:\ShareDisk\ipocDiskTrend\trend\trend'

def checkdir(*args):
    for d in args:
        if not os.path.exists(d):
            os.mkdir(d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', required=True, help='ipoc project env')
    parser.add_argument('-m', '--modeldir', default='./model', help='output model dir;default is ./')
    parser.add_argument('-i', '--imagedir', default='./imgReport',
                        help='output image dir;default is ./imgReport')
    args = parser.parse_args()
    #args.func(args)

    global modeldir
    modeldir = args.modeldir
    global reportdir
    reportdir = args.imagedir
    checkdir(modeldir,reportdir)

    bigdir = '/diskZ/big/'
    diskTrend2(os.path.join(bigdir,args.env,'trend'), histday='30d',sigma=3)


if __name__ == '__main__':
    main()
