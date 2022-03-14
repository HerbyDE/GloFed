import os


def verbose_debug_msg(msg, level=1):
    """
    This method returns a message based on global verbosity settings. Make sure the verbosity level can be coerced to
    int. If the setting is invalid, this function will fail silently.
    :param msg: Message string
    :param level: verbosity level: 0 no vervosity -> 4 full verbosity / explanations (if available).
    :return: None
    """
    try:
        verbosity = int(os.environ["DEBUG_VERBOSITY"])
    except ValueError:
        return

    if verbosity <= level:
        print(msg)

    return