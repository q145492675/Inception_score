# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 20:58:37 2018

@author: Herman Wu
"""

from six.moves import urllib
import os

def cbk(a,b,c):
    per=100*a*b/c
    if per>100:
        per=100
    print('%.2f%%',per)

if __name__ == '__main__':
    url='http://www.google.com'
    Base_dir = os.path.dirname(__file__)
    local=Base_dir+'/test-urlib'
    if not os.path.exists(local):
        os.makedirs(local,mode=0o777)
    urllib.request.urlretrieve(url,(local+'/test'),cbk)

def Schedule(a,b,c):
    '''''
    a:已经下载的数据块
    b:数据块的大小
    c:远程文件的大小
   '''
    per = 100.0 * a * b / c
    if per > 100 :
        per = 100
    print ('%.2f%%' , per)
url = 'http://www.python.org/ftp/python/2.7.5/Python-2.7.5.tar.bz2'
#local = url.split('/')[-1]
local = os.path.join('/data/software','Python-2.7.5.tar.bz2')
urllib.urlretrieve(url,local,Schedule)
######output######
#0.00%
#0.07%
#0.13%
#0.20%
#....
#99.94%
#100.00%
