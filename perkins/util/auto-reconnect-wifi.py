"""
Created by PerkinsZhu on 2022/8/18 12:40
"""

import os
import subprocess
from time import strftime, localtime

from apscheduler.schedulers.blocking import BlockingScheduler

ssid = "wifi name"


def reconnect():
    '''
    重连
    '''
    print("%s 正在重连WiFi" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
    os.system("netsh wlan disconnect")
    os.system("netsh wlan connect ssid=%s name=%s" % (ssid, ssid))


def check_wifi():
    subp = subprocess.Popen("ping baidu.com", stdout=subprocess.PIPE)
    while subp.poll() is None:
        line = subp.stdout.readline()
        text = str(line, encoding='GBK')
        print(" %s" % text)
        if match(text):
            reconnect()
            break


def match(text):
    '''
    匹配
    '''
    if text:
        if text.find('失败') >= 0 or text.find('请求找不到主机') >= 0:
            return True
        else:
            return False
    else:
        return False


if __name__ == '__main__':
    print("启动定时器任务，每60秒执行一次")
    s = BlockingScheduler()
    s.add_job(check_wifi, 'interval', seconds=60, timezone='Asia/Shanghai')  # 10秒运行一次
    s.start()
    print("任务启动成功")
