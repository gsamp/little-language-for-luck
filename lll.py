import random

class trial:
    def __init__(self):
        self.cache = {}

    def __getitem__(self, obj):
        if not isinstance(obj, (random_variable, event)):
            return obj
        if obj not in self.cache:
            self.cache[obj] = obj.compute(self)
        return self.cache[obj]


class random_variable:
    """
    A real-valued random variable
    """

    def __init__(self, compute=None):
        self.compute = compute

    def __num__(self):
        raise SyntaxError('You cannot directly use a random variable as a number')

    def __bool__(self):
        raise SyntaxError('You cannot directly use a random variable as a bool')

    def __add__(self, other):
        return random_variable(lambda trial: trial[self] + trial[other])

    def __radd__(self, other):
        return random_variable(lambda trial: trial[other] + trial[self])

    def __sub__(self, other):
        return random_variable(lambda trial: trial[self] - trial[other])

    def __rsub__(self, other):
        return random_variable(lambda trial: trial[other] - trial[self])

    def __mul__(self, other):
        return random_variable(lambda trial: trial[self] * trial[other])

    def __rmul__(self, other):
        return random_variable(lambda trial: trial[other] * trial[self])

    def __floordiv__(self, other):
        return random_variable(lambda trial: trial[self] // trial[other])

    def __rfloordiv__(self, other):
        return random_variable(lambda trial: trial[other] // trial[self])

    def __truediv__(self, other):
        return random_variable(lambda trial: trial[self] / trial[other])

    def __rtruediv__(self, other):
        return random_variable(lambda trial: trial[other] / trial[self])

    def __pow__(self, other):
        return random_variable(lambda trial: trial[self] ** trial[other])

    def __rpow__(self, other):
        return random_variable(lambda trial: trial[other] ** trial[self])

    def __neg__(self, other):
        return random_variable(lambda trial: -trial[self])

    def __eq__(self, other):
        return event(lambda trial: trial[self] == trial[other])

    def __req__(self, other):
        return event(lambda trial: trial[other] == trial[self])

    def __ne__(self, other):
        return event(lambda trial: trial[self] != trial[other])

    def __rne__(self, other):
        return event(lambda trial: trial[other] != trial[self])

    def __gt__(self, other):
        return event(lambda trial: trial[self] > trial[other])

    def __rgt__(self, other):
        return event(lambda trial: trial[other] > trial[self])

    def __ge__(self, other):
        return event(lambda trial: trial[self] >= trial[other])

    def __rge__(self, other):
        return event(lambda trial: trial[other] >= trial[self])

    def __lt__(self, other):
        return event(lambda trial: trial[self] < trial[other])

    def __rlt__(self, other):
        return event(lambda trial: trial[other] < trial[self])

    def __le__(self, other):
        return event(lambda trial: trial[self] <= trial[other])

    def __rle__(self, other):
        return event(lambda trial: trial[other] <= trial[self])

    def __hash__(self):
        return hash(self.compute)


class event:
    def __init__(self, compute):
        self.compute = compute

    def __bool__(self):
        raise SyntaxError('You cannot directly use an event as a bool; use bitwise operations instead.')

    def __or__(self, other):
        return event(lambda trial: trial[self] | trial[other])

    def __ror__(self, other):
        return event(lambda trial: trial[other] | trial[self])

    def __and__(self, other):
        return event(lambda trial: trial[self] & trial[other])

    def __rand__(self, other):
        return event(lambda trial: trial[other] & trial[self])

    def __invert__(self):
        return event(lambda trial: ~trial[self])

    def if_then_else(self, t, f):
        return random_variable(lambda trial: trial[t] if trial[self] else trial[f])

    def binary_indicator(self):
        return self.if_then_else(1, 0)



class distribution:
    def __init__(self):
        pass

    def __invert__(self):  # ~ operator
        return self.draw()

    def draw(self):
        raise NotImplemented

class Integrator:
    def integrate(self, rvs, condition, sampling):
        s = [0 for _ in rvs]
        n = 0
        for i in range(sampling):
            t = trial()
            if t[condition]:
                s = [a + t[rv] for a, rv in zip(s, rvs)]
                n += 1
        return [a / n for a in s]

    def __getitem__(self, subscript):
        rv = subscript
        condition = True
        sampling = None
        if isinstance(subscript, slice):
            rv = subscript.start
            condition = subscript.stop
            sampling = subscript.step

        if sampling is None:
            sampling = 10_000

        if condition is None:
            condition = True

        return self.compute(rv, condition, sampling)

    def compute(self, rv, condition, sampling):
        raise NotImplemented

class Expectation(Integrator):
    def compute(self, rv, condition, sampling):
        return self.integrate([rv], condition, sampling)[0]

class Variance(Integrator):
    def compute(self, rv, condition, sampling):
        ex, ex2 = self.integrate([rv, rv ** 2], condition, sampling)
        return (ex2 - ex ** 2) * sampling / (sampling - 1)

class Probability(Integrator):
    def compute(self, rv, condition, sampling):
        return self.integrate([rv.if_then_else(1, 0)], condition, sampling)[0]

E = Expectation()
Pr = Probability()
Var = Variance()


class Bernoulli(distribution):
    def __init__(self, p):
        self.p = p
    
    def draw(self):
        return event(
            lambda trial: random.random() < trial[self.p]
        )

class Uniform(distribution):
    def __init__(self, xs):
        self.xs = xs

    def draw(self):
        return random_variable(
            lambda trial: trial[random.choice(self.xs)]
        )


class Die(distribution):
    def __init__(self, N):
        self.N = N

    def draw(self):
        return random_variable(
            lambda trial: random.randint(1, trial[self.N])
        )


if __name__ == '__main__':

    # A small problem taken from CS109 Fall 2020 Pset 2
    is_robot = ~ Bernoulli(0.05)
    P_pass_captcha = is_robot.if_then_else(0.3, 0.95)

    captchas = []
    for i in range(5):
        captchas.append(~ Bernoulli(P_pass_captcha))

    score = sum([c.binary_indicator() for c in captchas])
    is_flagged = (score < 5)

    print('Robot flagged w/ prob', Pr[is_flagged :  is_robot])
    print('Human flagged w/ prob', Pr[is_flagged : ~is_robot])
    print('Flag is robot w/ prob', Pr[is_robot   :  is_flagged])


    # A "standard" Bayes' Rule problem that I still have *no* intuition for...
    has_disease = ~ Bernoulli(0.01)
    test_positive = has_disease.if_then_else(
        ~ Bernoulli(0.99),
        ~ Bernoulli(0.01) # false positives!
    )
    print(Pr[has_disease : test_positive : 100])

    N  = ~ Uniform([6])
    d1 = ~ Die(N)
    d2 = ~ Die(N)
    s = d1 + d2

    print(E[s], '+/-', Var[s] ** 0.5)
    print(Pr[d1 > 5 : d1 > 1])


    # Does the central limit theorem hold for 100 coin tosses?
    S = sum([(~Bernoulli(0.5)).if_then_else(1, 0) for _ in range(100)])
    mu = E[S]
    sigma = Var[S] ** 0.5
    print(Pr[(S - mu) / sigma > 1])
