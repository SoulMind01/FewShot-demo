import train
from train import *
from arguments import ARGUMENTS
import os
import numpy as np
import matplotlib.pyplot as plt
from arguments import reset_args
from tqdm import tqdm

google_drive_path = "/content/drive/MyDrive/FewShot-demo"


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


def analyze_averaged_results(
    results: dict,
    x_axis: list,
    feature_name: str,
    x_log_scale: bool = False,
    figure_size: tuple = (16, 4),
):
    plt.figure(figsize=figure_size)
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
                value = [np.mean(value[key]) for key in value]
                plt.plot(x_axis, value, label=key, marker=markers.pop(0))
                plt.xscale("log")
            else:
                value = [np.mean(value[key]) for key in value]
                plt.plot(x_axis, value, label=key, marker=markers.pop(0))
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.subplots_adjust(wspace=0.35, hspace=0.7)


def transpose_results_dimension(results: dict, class_range: np.ndarray) -> dict:
    # original: results[dataset][stat][feature_value]=[value0, value1, ..., valueN]
    # new: results[dataset][stat][class]=[v_feature0, v_feature1, ..., v_featureN]
    new_results = {}
    for dataset in results.keys():
        new_results[dataset] = {}
        for stat in results[dataset].keys():
            new_results[dataset][stat] = {}
            for class_ in range(len(class_range)):
                new_results[dataset][stat][class_] = []
                if stat == "df":
                    continue
                for feature_value in results[dataset][stat].keys():
                    new_results[dataset][stat][class_].append(
                        results[dataset][stat][feature_value][class_]
                    )
    return new_results


def analyze_results_with_box(
    results: dict,
    x_axis: list,
    experiment_name: str,
    feature_name: str,
    class_range: np.ndarray,
    x_log_scale: bool = False,
    figure_size: tuple = (16, 5),
    labels: np.ndarray = None,
    box_width: float = 0.5,
    positions: np.ndarray = None,
):
    colors = {
        "auc": "blue",
        "f1": "green",
        "spec": "red",
        "recall": "purple",
        "acc": "orange",
    }
    # 5 different performances subplots for each dataset
    for dataset, dataset_results in results.items():
        plt.figure(figsize=figure_size)
        plt.suptitle(f"Performance along {feature_name} of {dataset} dataset")

        # remove key df from the results
        if "df" in dataset_results:
            dataset_results.pop("df")
        for key, value in dataset_results.items():
            # assign a subplot for each feature
            plt.subplot(1, 5, list(dataset_results.keys()).index(key) + 1)
            plt.subplots_adjust(wspace=0.15, hspace=0.7)
            plt.title(key)
            color = colors[key]

            if x_log_scale:
                plt.boxplot(
                    [value[key] for key in value],
                    patch_artist=True,
                    # labels=[key for key in value],
                    widths=box_width,
                    boxprops=dict(facecolor=color),
                    positions=labels,
                )
                plt.xscale("log")
            else:
                plt.boxplot(
                    [value[key] for key in value],
                    patch_artist=True,
                    # labels=([key for key in value] if labels is None else labels),
                    widths=box_width,
                    boxprops=dict(facecolor=color),
                    positions=labels,
                )
                # print the unit of the scaled x-axis
                if labels is not None:
                    dx = labels[1] - labels[0]
                    dfeature = x_axis[1] - x_axis[0]
                    ratio = dx / dfeature if dx > dfeature else dfeature / dx
                    if dx > dfeature:
                        plt.xlabel(f"{feature_name} (x{ratio})")
                    else:
                        plt.xlabel(f"{feature_name} (/{ratio})")
                else:
                    plt.xlabel(feature_name)
            plt.savefig(f"{experiment_name}/{feature_name}_experiment_{dataset}.png")


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
    datasets = ["cifar10", "fashion", "mnist"]
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
                for i in tmp:
                    results[dataset][feature][i] = tmp[i]
            else:
                results[dataset][feature] = tmp
    return results, is_widebox


def report_results(
    results: dict,
    experiment_name: str,
    feature_name: str,
    feature_range: list,
    class_range: np.ndarray,
    google_drive_path: str = None,
    box_plot: bool = False,
    figure_size: tuple = (16, 4),
    box_width: float = 0.5,
    specify_xticks: np.ndarray = None,
    save_path: str = "",
):
    if not os.path.exists(experiment_name):
        os.mkdir(experiment_name)
    analyze_results_with_box(
        results,
        feature_range,
        experiment_name,
        feature_name.capitalize(),
        class_range,
        x_log_scale=False,
        figure_size=figure_size,
        box_width=box_width,
        labels=specify_xticks,
    )
    analyze_averaged_results(
        results,
        feature_range,
        feature_name.capitalize(),
        x_log_scale=True if "earning" in feature_name else False,  # lr
        figure_size=figure_size,
    )
    plt.savefig(f"{experiment_name}/{experiment_name}.png")
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
    class_range: np.ndarray = np.arange(0, 10),
    num_test_data: int = 0,
    quick_run: bool = False,
    test_experiment: bool = False,
    boxplot: bool = True,
    figure_size: tuple = (20, 4),
    box_width: float = 0.5,
    specify_xticks: np.ndarray = None,
    lock_feature: bool = False,
):
    experiment_name = f"{experiment_name}_{args.evaluation_method}"
    i = 1
    results = init_result_dicts()
    if os.path.exists(experiment_name) and os.listdir(experiment_name):
        results, _ = load_results(experiment_name)
        print("Results loaded from file")
    else:
        for feature in tqdm(feature_range, desc=f"Running {experiment_name}"):
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
                    args.dataset_name = dataset
                    args.normal_class = class_
                    if not lock_feature:
                        setattr(args, feature_name, feature)
                    # print(f"dataset: {dataset}, feature: {feature}, class: {class_}")
                    if test_experiment:
                        # assign 6 random values to the feature
                        df, auc, f1, spec, recall, acc, _ = (
                            1 + class_ + np.random.random(),
                            2 + class_ + np.random.random(),
                            3 + class_ + np.random.random(),
                            4 + class_ + np.random.random(),
                            5 + class_ + np.random.random(),
                            6 + class_ + np.random.random(),
                            None,
                        )
                        i += 1
                    else:
                        df, auc, f1, spec, recall, acc, _ = train(args, num_test_data)

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
        class_range,
        google_drive_path,
        boxplot,
        figure_size=figure_size,
        box_width=box_width,
        specify_xticks=specify_xticks,
    )
