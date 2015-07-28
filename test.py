__author__ = 'jeremy'


def myfunction(a):
    return a * a


def myfunction2(coefficient=1, exponent=5):
    return coefficient * 2 ** exponent


if __name__ == "__main__":
    print myfunction(3)
    print myfunction2()
    print myfunction2(exponent=4)
    print myfunction2(exponent=4, coefficient=-3)
    print myfunction2(coefficient=-3, exponent=4)
    print myfunction2(1.5, 3)