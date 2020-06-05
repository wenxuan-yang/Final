import pandas as pd
import requests


def process_data(paper):
    lines = paper.splitlines()
    generate_category = False
    record_data = False
    categories = []
    data = []
    for each in lines:
        line = each.strip()
        if len(line) != 0:
            if line == 'The variables listed below, from left to right, ' \
                       'are:':
                generate_category = True
            elif line == '(Measurement standards are apparently those ' \
                         'listed in Benhke and Wilmore':
                generate_category = False
            elif generate_category:
                categories.append(line)
            if line == 'Principles of the Conditioning Process_, Allyn ' \
                       'and Bacon, Inc., Boston.':
                record_data = True
            elif line == 'Roger W. Johnson':
                record_data = False
            elif record_data:
                data.append(line.split())
    return pd.DataFrame(data, columns=categories)


def main():
    url = 'http://lib.stat.cmu.edu/datasets/bodyfat'
    data = process_data(requests.get(url).text)
    print(data)


if __name__ == '__main__':
    main()