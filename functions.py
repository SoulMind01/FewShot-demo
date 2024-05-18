import train
from train import *

google_drive_path = "/content/drive/MyDrive/FewShot-demo"


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
        dataset="fashion",
        distance_method="multi",
        model=VGG16(vector_size=1024, biases=1, dataset_name="cifar10"),
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
    args.dataset = "fashion"
    args.distance_method = "multi"
    return args


def analyze_result(
    results: dict,
    feature_name: str,
    selected_performances: list = None,
    x_axis: list = None,
):
    plt.figure(figsize=(10, 8))
    markers = [
        "o",
        "s",
        "D",
        "x",
        "^",
        "v",
        "<",
        ">",
        "p",
        "h",
        "+",
        "1",
        "2",
        "3",
        "4",
        "8",
        "s",
        "p",
        "*",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]
    for key, value in results.items():
        if key == "df" or "df" in key:
            continue
        if selected_performances is not None and key not in selected_performances:
            continue
        if x_axis is not None:
            plt.plot(x_axis, value, label=key, marker=markers.pop(0))
        else:
            plt.plot(value, label=key, marker=markers.pop(0))
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.title(f"Performance along {feature_name}")
    plt.grid()


def analyze_results(
    results: dict, x_axis: list, feature_name: str, x_log_scale: bool = False
):
    plt.figure(figsize=(16, 4))
    plt.suptitle(f"Performance along {feature_name}")
    markers = [
        "o",
        "s",
        "D",
        "x",
        "^",
        "v",
        "<",
        ">",
        "p",
        "h",
        "+",
        "1",
        "2",
        "3",
        "4",
        "8",
        "s",
        "p",
        "*",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]
    for dataset, dataset_results in results.items():
        plt.subplot(1, 3, list(results.keys()).index(dataset) + 1)
        plt.title(f"{dataset.capitalize()}")
        for key, value in dataset_results.items():
            if key == "df" or "df" in key:
                continue
            if x_log_scale:
                plt.semilogx(x_axis, value, label=key, marker=markers.pop(0))
            else:
                value = [np.mean(value[key]) for key in value]
                plt.plot(x_axis, value, label=key, marker=markers.pop(0))
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.subplots_adjust(wspace=0.5)


def analyze_results_with_box(
    results: dict, x_axis: list, feature_name: str, x_log_scale: bool = False
):
    # 5 different performances subplots for each dataset
    for dataset, dataset_results in results.items():
        plt.figure(figsize=(20, 4))
        plt.suptitle(f"Performance along {feature_name} of {dataset} dataset")

        # remove key df from the results
        if "df" in dataset_results:
            dataset_results.pop("df")
        for key, value in dataset_results.items():
            # assign a subplot for each feature
            plt.subplot(1, 5, list(dataset_results.keys()).index(key) + 1)
            plt.subplots_adjust(wspace=0.5)
            plt.title(key)
            if x_log_scale:
                # plt.semilogx(x_axis, value, label=key, marker=markers.pop(0))
                plt.boxplot(
                    [value[key] for key in value],
                    positions=x_axis,
                    patch_artist=True,
                    labels=[key for key in value],
                )
                plt.xscale("log")
            else:
                plt.boxplot(
                    [value[key] for key in value],
                    positions=x_axis,
                    patch_artist=True,
                    labels=[key for key in value],
                )


def plot_hist(results: dict):
    # select the best index according to the f1 score
    best_index = np.argmax(results["f1"])
    tmp_df = results["df"][best_index]
    plt.figure(figsize=(10, 8))
    thres = np.percentile(tmp_df["minimum_dists"], 10)
    for label in tmp_df["label"].unique():
        plt.hist(
            tmp_df[tmp_df["label"] == label]["minimum_dists"],
            alpha=0.5,
            label=label,
            bins=100,
        )
    plt.axvline(x=thres, color="r", linestyle="--")
    plt.ylabel("Count")
    plt.xlabel("Anomaly Score")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.title("Anomaly Score Distribution")
    plt.grid()


def init_result_dicts():
    features = ["df", "auc", "f1", "spec", "recall", "acc"]
    datasets = ["fashion", "mnist", "cifar10"]
    result_dicts = {}
    for dataset in datasets:
        result_dicts[dataset] = {}
        for feature in features:
            result_dicts[dataset][feature] = {}
    return result_dicts


def save_results(results: dict, save_path: str):
    for dataset in results.keys():
        for stat in results[dataset].keys():
            if stat == "df":
                continue
            if not os.path.exists(f"{save_path}/{dataset}"):
                os.mkdir(f"{save_path}/{dataset}")
            np.save(f"{save_path}/{dataset}/{stat}.npy", results[dataset][stat])


# return result and whether it is a widebox result or not
def load_results(results_path: str) -> tuple[dict, bool]:
    results = {}
    for dataset in ["fashion", "mnist", "cifar10"]:
        results[dataset] = {}
        for feature in ["auc", "f1", "spec", "recall", "acc"]:
            tmp = np.load(
                f"{results_path}/{dataset}/{feature}.npy", allow_pickle=True
            ).item()
            # judge whether tmp is a widebox result or not
            is_widebox = len(tmp) != 1
            if is_widebox:
                results[dataset][feature] = {}
                for i, key in enumerate(tmp):
                    results[dataset][feature][key] = tmp[i]
            else:
                results[dataset][feature] = tmp
    return results, is_widebox


def report_results(
    results: dict,
    experiment_name: str,
    feature_name: str,
    feature_range: list,
    google_drive_path: str = None,
    box_plot: bool = False,
):
    if not os.path.exists(experiment_name):
        os.mkdir(experiment_name)
    if box_plot:
        analyze_results_with_box(
            results,
            feature_range,
            feature_name.capitalize(),
            x_log_scale=True if "earning" in feature_name else False,  # lr
        )
    else:
        analyze_results(
            results,
            feature_range,
            feature_name.capitalize(),
            x_log_scale=True if "earning" in feature_name else False,  # lr
        )
    plt.savefig(f"{experiment_name}/{feature_name}_experiment.png")
    save_results(results, experiment_name)

    if is_colab():
        os.system(f"cp -r {experiment_name} {google_drive_path}")


def is_colab():
    for key in os.environ.keys():
        if "COLAB" in key:
            return True
    return False


def do_experiment(
    args: ARGUMENTS,
    experiment_name: str,
    feature_name: str,
    feature_range: list,
    num_test_data: int = 0,
    class_range: list = np.arange(0, 10, 1),
    quick_run: bool = False,
    test_experiment: bool = False,
    boxplot: bool = False,
):
    results = init_result_dicts()
    if os.path.exists(experiment_name):
        results, _ = load_results(experiment_name)
        print("Results loaded from file")
    else:
        for feature in feature_range:
            for dataset in results.keys():
                best_df = None
                best_auc = 0
                results[dataset]["auc"][feature] = []
                results[dataset]["f1"][feature] = []
                results[dataset]["spec"][feature] = []
                results[dataset]["recall"][feature] = []
                results[dataset]["acc"][feature] = []
                for class_ in class_range:
                    reset_args(args)
                    if not "class" in feature_name:
                        args.normal_class = class_
                    args.dataset = dataset
                    setattr(args, feature_name, feature)
                    if test_experiment:
                        # assign 6 random values to the feature
                        df, auc, f1, spec, recall, acc = np.random.rand(6) * 3
                    else:
                        df, auc, f1, spec, recall, acc = train(args, num_test_data)

                    if quick_run:
                        return

                    best_df = df if auc > best_auc else best_df
                    best_auc = auc if auc > best_auc else best_auc

                    results[dataset]["auc"][feature].append(auc)
                    results[dataset]["f1"][feature].append(f1)
                    results[dataset]["spec"][feature].append(spec)
                    results[dataset]["recall"][feature].append(recall)
                    results[dataset]["acc"][feature].append(acc)
                results[dataset]["df"] = best_df

    report_results(
        results,
        experiment_name,
        feature_name,
        feature_range,
        google_drive_path,
        boxplot,
    )
