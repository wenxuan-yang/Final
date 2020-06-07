import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


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


def process_another_data(data, columns):
    columns_lines = columns.splitlines()
    generate_category = False
    record_data = False
    data
    categories = []
    for each in columns_lines:
        line = each.strip()
        if len(line) != 0:
            if line == 'Columns     Variable':
                generate_category = True
            elif line == "'</PRE>', '</P>', '<P>'":
                generate_category = False
            elif generate_category:
                categories.append(line)
    print(categories)


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
    plt.savefig('test.png')


def linear_regression_fit(x, y):
    reg = LinearRegression().fit(x, y)
    print('Linear Regression score:', reg.score(x, y))
    # print('Linear Regression coefficient:', reg.coef_)
    # print('Linear Regression intercept:', reg.intercept_)
    return reg


def main():
    sns.set(font_scale=0.7)
    url = 'http://lib.stat.cmu.edu/datasets/bodyfat'
    url_another_data = 'http://jse.amstat.org/datasets/body.dat.txt'
    url_another_columns = 'http://jse.amstat.org/v11n2/datasets.heinz.html'
    process_another_data(requests.get(url_another_data).text,
                         requests.get(url_another_columns).text)
    data, columns = process_data(requests.get(url).text)
    # url_another_data = 'http://jse.amstat.org/datasets/body.dat.txt'
    calculate_bmi(data)
    # print(correlation_chart(data, columns))
    print('Correlation coefficient:', correlation(data, 'BMI'))
    graphs(data)
    x = data.drop(["Percent body fat from Siri's (1956) equation"], axis=1).to_numpy()
    y = data["Percent body fat from Siri's (1956) equation"].to_numpy()
    # train : dev : test = 7 : 1.5 : 1.5
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    x_dev, x_test, y_dev, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=1)
    # Linear Regression
    linear_reg = linear_regression_fit(x_train, y_train)
    print('MSE for train:', mean_squared_error(y_train, linear_reg.predict(x_train)))
    print('MSE for dev  :', mean_squared_error(y_dev, linear_reg.predict(x_dev)))


if __name__ == '__main__':
    main()
