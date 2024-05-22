class ARGUMENTS:
    def __init__(
        self,
        normal_class: int,
        num_ref: int,
        num_ref_eval: int,
        lr: float,
        vector_size: int,
        weight_decay: float,
        seed: int,
        weight_init_seed: int,
        alpha: float,
        k: int,
        epochs: int,
        contamination: float,
        batch_size: int,
        biases: int,
        dataset_name: str,
        distance_method: str,
        evaluation_method: str = None,
        activation_function: str = "leaky_relu",
    ):
        self.normal_class = normal_class
        self.num_ref = num_ref
        self.num_ref_eval = num_ref_eval
        self.lr = lr
        self.vector_size = vector_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.weight_init_seed = weight_init_seed
        self.alpha = alpha
        self.k = k
        self.epochs = epochs
        self.contamination = contamination
        self.batch_size = batch_size
        self.biases = biases
        self.dataset_name = dataset_name
        self.distance_method = distance_method
        self.evaluation_method = evaluation_method
        self.activation_function = activation_function


def init_args() -> ARGUMENTS:
    args = ARGUMENTS(
        normal_class=7,
        num_ref=10,
        num_ref_eval=5,  # size of reference set while testing
        lr=1e-5,
        vector_size=1024,
        weight_decay=0.001,  # done
        seed=42,
        weight_init_seed=42,
        alpha=0.15,  # done
        k=3,  # size of reference set while training
        epochs=10,
        contamination=0,  # done
        batch_size=1,
        biases=1,
        dataset_name="fashion",
        distance_method="multi",
        evaluation_method="anomaly_score",
    )
    return args


def reset_args(args: ARGUMENTS) -> ARGUMENTS:
    args.normal_class = 7
    args.num_ref = 10
    args.num_ref_eval = 5
    args.lr = 1e-5
    args.vector_size = 1024
    args.weight_decay = 0.001
    args.seed = 42
    args.weight_init_seed = 42
    args.alpha = 0.15
    args.k = 3
    args.epochs = 10
    args.contamination = 0
    args.batch_size = 1
    args.biases = 1
    return args
