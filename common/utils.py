import os
import errno


def create_dirs(path):
    if not os.path.exists(path):
        try:
            if os.path.isdir(path):
                os.makedirs(path)
            elif os.path.isfile(path):
                os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
