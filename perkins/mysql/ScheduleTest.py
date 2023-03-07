"""
Created by PerkinsZhu on 2022/6/21 15:58
"""

import schedule
import time
import logging

from datetime import datetime
import os
from apscheduler.schedulers.blocking import BlockingScheduler



def do_func():
    logging.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " in do func ...")
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " in do func ...")


def main():
    schedule.every(2).seconds.do(do_func)

    while True:
        schedule.run_pending()


def simpleSchedule():
    logging.basicConfig(format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG)
    logging.debug("===main===start")
    main();
    logging.debug("===main===end")



def tick():
    print('Tick! The time is: %s' % datetime.now())


def aPScheule():
    scheduler = BlockingScheduler()
    scheduler.add_job(tick, 'interval', seconds=3)
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))
    scheduler.start()


if __name__ == "__main__":
    # simpleSchedule()
    aPScheule()

