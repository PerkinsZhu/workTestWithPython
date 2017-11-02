'''
 Created by PerkinsZhu on 2017/10/28 14:19. 
'''
from sklearn import datasets as ds
from sklearn import svm
import pickle
def test01():
    iris = ds.load_iris()
    digits = ds.load_digits()
    print(digits.data)
    print(digits.target)
    print(digits.images[0])
    clf = svm.SVC(gamma=0.001,C=10)
    temp = clf.fit(digits.data[:-1],digits.target[:-1])
    print(temp)
    print(clf.predict(digits.data[-1:]))

    # s = pickle.dump(clf)
    # clf2 = pickle.load(s)
    # clf2.predict[X[0]]

    def test02():
        a = arange(5)

if __name__ == '__main__':
    # test01()
    test02()
