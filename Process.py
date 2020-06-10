"""
Yutian Lei, Wenxuan Yang, Yining Liu
CSE163
This program uses several libraries to analyzes
and predicts the body fat data with machine learning.
We scrap data from target website and use data for
machine learning model training. Moreover, we plot
related data for our data analyzing.
"""


from read_data import DataReader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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
    Returns a list of tuples that has all the
    coefficient correlation of all the columns.
    :param data: The complete data.
    :param columns: The columns in the data.
    :return: The list that contains all the correlation coefficient
    in tuple from that corresponding the columns.
    """
    all_correlation = dict()
    temp_column = columns.copy()
    temp_column.remove("Percent body fat from Siri's (1956) equation")
    for each in temp_column:
        all_correlation[each] = correlation(data, each)
    all_correlation = sorted(all_correlation.items(), key=lambda x: x[1],
                             reverse=True)
    return all_correlation


def graphs(data, all_correlation):
    """
    Plots the data in png form
    :param data:
    :return:
    """
    sns.set_style("white")
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    temp = all_correlation.copy()
    print(temp.pop()[0])
    for i in range(4):
        for j in range(3):
            plotting = sns.scatterplot(
                x=temp.pop(0)[0],
                y="Percent body fat from Siri's (1956) "
                "equation", data=data, ax=axes[i, j]
            )
            plotting.set_ylabel('Percent Body Fat', fontsize=10)
    plt.ylabel('Percent Body Fat', fontsize=10)
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle('Body Measurements vs. Body Fat Percentage')
    plt.savefig('test.png', dpi=300)


def linear_regression_fit(x, y):
    """
    Calculate the score from given x and y dataset by using
    sklearn library
    :param x: dataset
    :param y: dataset
    :return:
    """
    model = LinearRegression().fit(x, y)
    print('Linear Regression score:', model.score(x, y))
    return model


def high_correlation(data):
    """
    Returns a list contains body parts that has high
    correlation coefficient value with percent body
    fat from the given data
    """
    correlation_list = []
    name = 'Density determined from underwater weighing'
    for pair in data:
        if (pair[1] >= 0.5) or (pair[1] <= -0.5):
            correlation_list.append(pair[0])
    if name in correlation_list:
        correlation_list.remove(name)
    return correlation_list


def main():
    sns.set(font_scale=0.7)
    # Process Data
    url = 'http://lib.stat.cmu.edu/datasets/bodyfat'
    web_data = DataReader(url)
    data, columns = web_data.read()
    # Plotting
    all_correlation = correlation_chart(data, columns)
    graphs(data, all_correlation)
    x = data.loc[:, 'Age (years)':'Wrist circumference (cm)']
    y = data["Percent body fat from Siri's (1956) equation"].to_numpy()
    # Linear Regression
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.4, random_state=1)
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    linear_reg_model = linear_regression_fit(x_train, y_train)
    print(
        'MSE for linear train:',
        mean_squared_error(y_train, linear_reg_model.predict(x_train))
    )
    print(
        'MSE for linear test:',
        mean_squared_error(y_test, linear_reg_model.predict(x_test))
    )
    print(
        'MSE for decisiontree train:',
        mean_squared_error(y_train, model.predict(x_train))
    )
    print(
        'MSE for decisiontree test:',
        mean_squared_error(y_test, model.predict(x_test))
    )
    # High correlation part
    x_high_correlation = data[high_correlation(all_correlation)].copy()
    x_high_train, x_high_test, y_high_train, y_high_test = \
        train_test_split(x_high_correlation, y, test_size=0.4, random_state=1)
    high_model = DecisionTreeRegressor()
    high_model.fit(x_high_train, y_high_train)
    high_correlation_model = linear_regression_fit(x_high_train, y_high_train)
    print(
        'MSE for high correlation train:',
        mean_squared_error(
            y_high_train, high_correlation_model.predict(x_high_train)
        )
    )
    print(
        'MSE for high correlation test:',
        mean_squared_error(
            y_high_test, high_correlation_model.predict(x_high_test)
        )
    )
    print(
        'MSE for high correlation decisiontree train:',
        mean_squared_error(y_high_train, high_model.predict(x_high_train))
    )
    print(
        'MSE for high correlation decisiontree test:',
        mean_squared_error(y_high_test, high_model.predict(x_high_test))
    )


if __name__ == '__main__':
    main()
