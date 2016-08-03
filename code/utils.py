import os
import errno
import configparser


def mkdirs(path):
    if not os.path.exists(path):
        try:
            if os.path.isfile(path):
                print(path)
                os.makedirs(os.path.dirname(path))
            else:
                print(path)
                os.makedirs(path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def readconf(file):
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation(),
                                       inline_comment_prefixes=(';',))
    config.read(file)

    params = {}

    for section in config.sections():
        for option in config[section]:
            params[option] = config.get(section, option)

    return params
