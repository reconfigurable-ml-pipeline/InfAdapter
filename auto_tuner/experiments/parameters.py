
class ParamTypes:
    CPU = "CPU"
    MEMORY = "MEM"
    REPLICA = "REPLICA"
    BATCH = "BATCH"
    BATCH_TIMEOUT = "BATCH_TIMEOUT"
    HARDWARE = "HARDWARE"
    MODEL_ARCHITECTURE = "ARCH"
    INTRA_OP_PARALLELISM = "INTRA_OP_PARALLELISM"
    INTER_OP_PARALLELISM = "INTER_OP_PARALLELISM"
    NUM_BATCH_THREADS = "NUM_BATCH_THREADS"
    MAX_ENQUEUED_BATCHES = "MAX_ENQUEUED_BATCHES"

    # hardware options
    HARDWARE_CPU = 1
    HARDWARE_GPU = 2

    @classmethod
    def get_all(cls):
        return [
            cls.CPU,
            cls.MEMORY,
            cls.REPLICA,
            cls.BATCH,
            cls.BATCH_TIMEOUT,
            # cls.HARDWARE,
            cls.MODEL_ARCHITECTURE,
            cls.INTRA_OP_PARALLELISM,
            cls.INTER_OP_PARALLELISM,
            cls.NUM_BATCH_THREADS,
            cls.MAX_ENQUEUED_BATCHES
        ]
