class BZ2Compressor:
    def __init__(self, compresslevel=9):
        raise NotImplementedError("bz2 is not available in this Python build")

    def compress(self, data):
        raise NotImplementedError("bz2 is not available in this Python build")

    def flush(self):
        raise NotImplementedError("bz2 is not available in this Python build")


class BZ2Decompressor:
    def __init__(self):
        raise NotImplementedError("bz2 is not available in this Python build")

    def decompress(self, data, max_length=-1):
        raise NotImplementedError("bz2 is not available in this Python build")

    @property
    def eof(self):
        return False

    @property
    def unused_data(self):
        return b""

    @property
    def needs_input(self):
        return True
