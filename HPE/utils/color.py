class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored_msg(msg, color=None):
    if color:
        return f"{getattr(Colors, color.upper())}{msg}\033[0m"
    else:
        return msg
