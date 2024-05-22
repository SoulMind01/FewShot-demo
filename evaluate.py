from distant_metrics import L1_dist, L2_dist, dist
from arguments import ARGUMENTS
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import train_test_split
from closest_dist_functions import (
    get_feature_embeddings,
    get_closest_class,
    construct_fewshot_dataloader,
)


def make_predictions_by_anormaly_score(
    args: ARGUMENTS,
    model: nn.Module,
    ref_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    anchor: torch.Tensor,
) -> tuple[pd.DataFrame, float, float, float, float, float, nn.Module]:
    """
    Train the model and make predictions based on the anomaly score.

    Parameters:
        args (ARGUMENTS): The arguments of the model.
        model (nn.Module): The model to train.
        ref_dataset (torch.utils.data.Dataset): The reference dataset.
        val_dataset (torch.utils.data.Dataset): The validation dataset.
        anchor (torch.Tensor): The anchor tensor.

    Returns:
        tuple[pd.DataFrame, float, float, float, float, float, nn.Module]: The predictions, the AUC, the F1 score, the specificity, the recall, the accuracy, and the model.
    """
    # device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_ref_eval = args.num_ref_eval
    model.eval()
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    query_distances = {}
    ref_embeddings = {}
    indexes = list(range(0, num_ref_eval))
    np.random.shuffle(indexes)

    for i in indexes:
        img, _, _, _ = ref_dataset.__getitem__(i)
        ref_embeddings[i] = model(img.to(device).float())
        query_distances[i] = []

    means = []
    minimum_dists = []
    labels = []

    with torch.inference_mode():
        for i, data in enumerate(loader):
            image = data[0][0]
            label = data[2].item()

            labels.append(label)
            total = 0
            min_dist = torch.Tensor([1e20])
            feature_embedding = model(image.to(device).float())

            dist_method = (
                dist
                if args.distance_method == "multi"
                else L1_dist if args.distance_method == "l1" else L2_dist
            )

            for j in range(num_ref_eval):
                distance = (
                    (1 - args.alpha)
                    * dist_method(feature_embedding, ref_embeddings[j])
                    / torch.sqrt(torch.Tensor([feature_embedding.size()[1]])).to(device)
                )
                +args.alpha * dist_method(feature_embedding, anchor) / torch.sqrt(
                    torch.Tensor([feature_embedding.size()[1]])
                ).to(device)

                query_distances[j].append(distance.item())
                total += distance.item()
                if distance.detach().item() < min_dist:
                    min_dist = distance.item()
            minimum_dists.append(min_dist)
            means.append(total)
    cols = ["label", "min_dist", "mean"]
    df = pd.concat(
        [
            pd.DataFrame(labels, columns=["label"]),
            pd.DataFrame(minimum_dists, columns=["min_dist"]),
            pd.DataFrame(means, columns=["mean"]),
        ],
        axis=1,
    )
    for i in range(num_ref_eval):
        df = pd.concat(
            [df, pd.DataFrame(query_distances[i], columns=[f"query_{i}"])], axis=1
        )
        cols.append(f"ref{i}")
    df.columns = cols
    df = df.sort_values(by="min_dist", ascending=False).reset_index(drop=True)

    preds = np.array(df["min_dist"])
    threshold = np.percentile(preds, 10)

    # make predictions
    (
        preds[preds > threshold],
        preds[preds <= threshold],
    ) = (1, 0)

    # get performance statistics
    fpr, tpr, _ = roc_curve(np.array(df["label"]), np.array(df["min_dist"]))
    auc = metrics.auc(fpr, tpr)
    auc = 0 if np.isnan(auc) else auc
    f1 = f1_score(np.array(df["label"]), preds)
    f1 = 0 if np.isnan(f1) else f1
    fp = len(df.loc[(preds == 1) & (df["label"] == 0)])
    tp = len(df.loc[(preds == 1) & (df["label"] == 1)])
    tn = len(df.loc[(preds == 0) & (df["label"] == 0)])
    fn = len(df.loc[(preds == 0) & (df["label"] == 1)])
    specficity = tn / (tn + fp) if tn + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
    print(
        f"auc: {auc:.4f}, f1: {f1:.4f}, spec: {specficity:.4f}, recall: {recall:.4f}, acc: {acc:.4f}"
    )
    return df, auc, f1, specficity, recall, acc, model


def make_predictions_by_closest_dist(
    args: ARGUMENTS,
    model: nn.Module,
    class_size: int = 50,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, float, float, float, float, float, nn.Module]:
    model = model.to("cpu")
    dataset_name = args.dataset_name
    image_num_per_class = [class_size] * 10
    assert len(image_num_per_class) == 10
    dataloader = construct_fewshot_dataloader(dataset_name, image_num_per_class)
    embeddings, labels = get_feature_embeddings(model, dataloader)
    train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
        embeddings,
        labels,
        test_size=test_ratio,
        random_state=args.seed,
    )
    # make query for every class
    pred_labels = []
    min_dists = []
    # assert all classes in the test set are in the train set
    assert np.all(np.isin(np.unique(test_labels), np.unique(train_labels)))
    for test_vector in test_embeddings:
        pred, min_dist = get_closest_class(
            embeddings=train_embeddings, labels=train_labels, query=test_vector
        )
        pred_labels.append(pred)
        min_dists.append(min_dist)

    # mark label to 0 if label = args.normal_class else 1
    pred_labels = np.array(pred_labels)
    test_labels = np.array(test_labels)
    (
        pred_labels[pred_labels != args.normal_class],
        pred_labels[pred_labels == args.normal_class],
    ) = (1, 0)
    (
        test_labels[test_labels != args.normal_class],
        test_labels[test_labels == args.normal_class],
    ) = (1, 0)

    # get performance statistics
    fpr, tpr, _ = roc_curve(test_labels, pred_labels)
    auc = metrics.auc(fpr, tpr)
    auc = 0 if np.isnan(auc) else auc
    f1 = f1_score(test_labels, pred_labels)
    f1 = 0 if np.isnan(f1) else f1
    fp = len(np.where((pred_labels == 1) & (test_labels == 0))[0])

    tp = len(np.where((pred_labels == 1) & (test_labels == 1))[0])
    tn = len(np.where((pred_labels == 0) & (test_labels == 0))[0])
    fn = len(np.where((pred_labels == 0) & (test_labels == 1))[0])
    specficity = tn / (tn + fp) if tn + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
    print(
        f"auc: {auc:.4f}, f1: {f1:.4f}, spec: {specficity:.4f}, recall: {recall:.4f}, acc: {acc:.4f}"
    )

    df = pd.DataFrame(
        {
            "label": test_labels,
            "pred": pred_labels,
            "min_dist": min_dists,
        }
    )
    return df, auc, f1, specficity, recall, acc, model
