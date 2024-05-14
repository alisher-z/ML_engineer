import numpy as np


def randomize(X, Y: np.ndarray):
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation, :]
    Y2 = [permutation]

    return X2, Y2


def draw_learning_curves(X, y, estimator, num_trainings):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt

    # X2, y2 = randomize(X, y)

    train_sizes, train_score, test_score = learning_curve(  # type: ignore
        estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1., num_trainings))

    train_score_mean = np.mean(train_score, axis=1)
    # train_score_std = np.std(train_score, axis=1)
    test_score_mean = np.mean(test_score, axis=1)
    # test_score_std = np.std(test_score, axis=1)

    print('numpy linspace')
    print(np.linspace(.1, 1., num_trainings))
    print('----------------------------------------------------------------------')
    print('train sizes')
    print(train_sizes)
    print('----------------------------------------------------------------------')
    print('train scores')
    print(train_score)
    print('----------------------------------------------------------------------')
    print('test scores')
    print(test_score)

    plt.grid()

    plt.title('Learning Curves')
    plt.xlabel('Training examples')
    plt.ylabel('Score')

    plt.plot(train_score_mean, 'o-', color="g", label="Training score")
    plt.plot(test_score_mean, 'o-', color="y", label='Cross-validation score')

    plt.legend(loc='best')

    plt.show()
