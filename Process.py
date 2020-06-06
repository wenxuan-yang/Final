import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns


def process_data(paper):
    """
    Returns a DataFrame to represent the data obtained from the paper.
    :param paper: The paper obtained online.
    :return: The DataFrame containing all the data in the paper.
    """
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
    processed_data = pd.DataFrame(data, columns=categories)
    processed_data = processed_data.astype(float)
    return processed_data, categories


def correlation(data, item):
    """
    Returns the correlation coefficient base on the given column and the body
    fat.
    :param data: The complete data set.
    :param item: The column that would be used to compute.
    :return: The correlation coefficient of the provided column and body fat.
    """
    x_mean = data[item].mean()
    y_mean = data["Percent body fat from Siri's (1956) equation"].mean()
    sxy_abbreviated = ((data[item] - x_mean) *
                       (data["Percent body fat from Siri's (1956) equation"]
                        - y_mean)).sum()
    sax_abbreviated = ((data[item] - x_mean)**2).sum()
    syy_abbreviated = ((data["Percent body fat from Siri's "
                             "(1956) equation"] - y_mean)**2).sum()
    r = sxy_abbreviated / np.sqrt(sax_abbreviated) / np.sqrt(syy_abbreviated)
    return r


def correlation_chart(data, columns):
    """
    Returns a dictionary that has all the coefficient correlation of all the
    columns.
    :param data: The complete data.
    :param columns: The columns in the data.
    :return: The dictionary that contains all the correlation coefficient
    corresponding the columns.
    """
    all_correlation = dict()
    temp_column = columns.copy()
    temp_column.remove("Percent body fat from Siri's (1956) equation")
    for each in temp_column:
        all_correlation[each] = correlation(data, each)
    all_correlation = sorted(all_correlation.items(), key=lambda x: x[1],
                             reverse=True)
    return all_correlation


def calculate_bmi(data):
    data['BMI'] = data['Weight (lbs)'] / data['Height (inches)']**2


def graphs(data):
    """
    Plots the data
    :param data:
    :return:
    """
    sns.set_style("white")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    sns.scatterplot(x='Abdomen 2 circumference (cm)',
                    y="Percent body fat from Siri's (1956) "
                         "equation", data=data, ax=ax1)
    sns.scatterplot(x='Chest circumference (cm)',
                    y="Percent body fat from Siri's (1956) "
                      "equation", data=data, ax=ax2)
    sns.scatterplot(x='Hip circumference (cm)',
                    y="Percent body fat from Siri's (1956) "
                      "equation", data=data, ax=ax3)
    sns.scatterplot(x='Density determined from underwater weighing',
                    y="Percent body fat from Siri's (1956) "
                      "equation", data=data, ax=ax4)
    plt.show()
    plt.savefig('test.png')


def main():
    sns.set(font_scale=0.7)
    url = 'http://lib.stat.cmu.edu/datasets/bodyfat'
    data, columns = process_data(requests.get(url).text)
    calculate_bmi(data)
    print(correlation_chart(data, columns))
    print(correlation(data, 'BMI'))
    graphs(data)


if __name__ == '__main__':
    main()