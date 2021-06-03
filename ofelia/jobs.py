import os
import sys
from datetime import datetime


def update_google_map_reviews():
    os.system('python map_review_crawler.py -p 王品牛排 -r 王品牛排')

def tagging():
    os.system('python tagging.py')

def analysis():
    os.system('python analysis.py')

def update_all():
    update_google_map_reviews()
    tagging()
    analysis()


if __name__ == '__main__':
    update_all()
