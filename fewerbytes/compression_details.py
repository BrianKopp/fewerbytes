class CompressionDetails:
    """
    Class which stores all the options and information required
    to decompress a data-set.
    """
    def __init__(self):
        """
        Initializes compression details. Set types through
        static pseudo-constructor methods
        """
        # uncompressed data shape
        self._bytes_per_value_uncompressed = 0

        # compressed data shape
        self._bytes_per_value_compressed = 0
        return
