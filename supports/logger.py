class logger:
    def __init__(self):
        self.last_len = 0
        self.str_line = None
        return

    def set_log(self, str_line: str):
        self.str_line = str_line

    def print(self, *args):
        print('\b' * self.last_len,end='')
        print_str = self.str_line % args
        print(print_str,end='')
        self.last_len = len(print_str)


if __name__ == "__main__":
    import time

    this_logger = logger()
    this_logger.set_log("hello world! seconds: %d")
    for i in range(100):
        this_logger.print(i + 1)
        time.sleep(1)
