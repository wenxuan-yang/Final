"""
Yutian Lei
CSE163 AD
This program test the python coding function
from two other file hw2_manual and hw2_pandas
with pandas library
"""

import pandas as pd

import requests

from cse163_utils import assert_equals, parse

import Process

def test_process_data(data, columns):
    assert_equals(252, len(data))
    assert_equals(15, len(columns))
    assert_equals(23, data.loc[0, 'Age (years)'])
    assert_equals('Weight (lbs)', columns[3])


def test_correlation(data):
    assert_equals(0.62520091, Process.correlation(data, 'Hip circumference (cm)'))
    assert_equals(0.29145844, Process.correlation(data, 'Age (years)'))
    assert_equals(-0.0894953, Process.correlation(data, 'Height (inches)'))

def test_correlation_chart(data, columns):
    test_list = [
        ('Abdomen 2 circumference (cm)', 0.813432284781049),
        ('Chest circumference (cm)', 0.7026203388938643),
        ('Hip circumference (cm)', 0.6252009175086622),
        ('Weight (lbs)', 0.6124140022026474),
        ('Thigh circumference (cm)', 0.5596075319940894),
        ('Knee circumference (cm)', 0.5086652428854677),
        ('Biceps (extended) circumference (cm)', 0.49327112589161576),
        ('Neck circumference (cm)', 0.49059185344104),
        ('Forearm circumference (cm)', 0.3613869031997191),
        ('Wrist circumference (cm)', 0.346574864526586),
        ('Age (years)', 0.29145844013522193),
        ('Ankle circumference (cm)', 0.26596977030637325),
        ('Height (inches)', -0.08949537985440179),
        ('Density determined from underwater weighing', -0.9877824021639865)
    ]
    assert_equals(test_list, Process.correlation_chart(data, columns))


def main():
    url = 'http://lib.stat.cmu.edu/datasets/bodyfat'
    data, columns = Process.process_data(requests.get(url).text)
    test_process_data(data, columns)
    test_correlation(data)
    test_correlation_chart(data, columns)


if __name__ == '__main__':
    main()