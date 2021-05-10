from minisom import MiniSom as ms


def somCluster(x, input_len, row=20, col=20, sigma=1.0, learning_rate=0.2, num_iter=500):
    som = ms(x=row, y=col, input_len=input_len, sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(x)
    som.train_random(data=x, num_iteration=num_iter)

    distance_map = som.distance_map().round(1)
    nodes = som.win_map(x)
    nonOutliers = []
    for i in range(row):
        for j in range(col):
            if distance_map[i, j] < 0.8:
                for element in nodes[i, j]:
                    nonOutliers.append(element)

    return nonOutliers
