from distant_metrics import L1_dist, L2_dist, dist
from arguments import ARGUMENTS
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, roc_curve


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
    f1 = f1_score(np.array(df["label"]), preds)
    fp = len(df.loc[(preds == 1) & (df["label"] == 0)])
    tp = len(df.loc[(preds == 1) & (df["label"] == 1)])
    tn = len(df.loc[(preds == 0) & (df["label"] == 0)])
    fn = len(df.loc[(preds == 0) & (df["label"] == 1)])
    specficity = tn / (tn + fp)
    recall = tp / (tp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    print(
        f"auc: {auc:.4f}, f1: {f1:.4f}, spec: {specficity:.4f}, recall: {recall:.4f}, acc: {acc:.4f}"
    )
    return df, auc, f1, specficity, recall, acc, model
