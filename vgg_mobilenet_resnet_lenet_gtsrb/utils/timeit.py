import time


def timeit(ds, steps=5000):
    start = time.time()
    it = iter(ds)
    for i in range(steps):
        _ = next(it)
        if i % 50 == 0:
            print('.', end='')
    print()
    end = time.time()

    duration = end - start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} samples/s".format(256 * steps / duration))
