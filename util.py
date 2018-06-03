from time import time

# utility to evaluate time
def tictoc(info):
    def _tictoc(fn):
        def wrap(*args, **kwargs):
            print(info, '...')
            tic = time()
            ret = fn(*args, **kwargs)
            elapsed = time() - tic
            print('finished in {:.2f}ms'.format(elapsed * 1000))
            return ret
        return wrap
    return _tictoc
