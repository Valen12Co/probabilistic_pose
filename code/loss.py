import torch
from typing import Union

from utils.tic import get_positive_definite_matrix, get_tic_covariance
from models.vit_pose.ViTPose import ViTPose
from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass


def mse_loss(means: torch.Tensor, gt:torch.Tensor) -> torch.Tensor:
    visible_joints = (gt[:, 0, :, 2] >= -0.5).int()
    loss = torch.zeros(means.size(0))
    for i, batch_idx in enumerate(range(means.size(0))):
        visible_indices = visible_joints[batch_idx].nonzero(as_tuple=True)[0]  # Indices of True values
        cov_indices = torch.cat([visible_indices * 2, visible_indices * 2 + 1]).sort()[0]
        mean = means[batch_idx][cov_indices].unsqueeze(0)
        loss[i] = (mean ** 2).sum(dim=1)
    return loss.mean()


def nll_loss(means: torch.Tensor, matrix: torch.Tensor, dim: int, gt: torch.Tensor) -> torch.Tensor:
    precision_hat = get_positive_definite_matrix(matrix, dim)
    visible_joints = (gt[:, 0, :, 2] >= -0.5).int()
    new_loss = torch.zeros(precision_hat.size(0))
    for i, batch_idx in enumerate(range(precision_hat.size(0))):
        visible_indices = visible_joints[batch_idx].nonzero(as_tuple=True)[0]  # Indices of True values
        cov_indices = torch.cat([visible_indices * 2, visible_indices * 2 + 1]).sort()[0]
        filtered_cov = precision_hat[batch_idx][cov_indices][:,cov_indices].unsqueeze(0)
        mean = means[batch_idx][cov_indices].unsqueeze(0)
        new_loss[i] = -torch.logdet(filtered_cov) + torch.matmul(
            torch.matmul(mean.unsqueeze(1),
                          filtered_cov),mean.unsqueeze(2)).squeeze()
    loss = -torch.logdet(precision_hat) + torch.matmul(
        torch.matmul(means.unsqueeze(1), precision_hat),
        means.unsqueeze(2)).squeeze()
    
    return new_loss.mean()


def diagonal_loss(means: torch.Tensor, matrix: torch.Tensor, dim: int, gt: torch.Tensor) -> torch.Tensor:
    var_hat = matrix[:, :dim] ** 2

    visible_joints = (gt[:, 0, :, 2] >= -0.5).int()
    new_loss = torch.zeros(var_hat.size(0))
    for i, batch_idx in enumerate(range(var_hat.size(0))):
        visible_indices = visible_joints[batch_idx].nonzero(as_tuple=True)[0]  # Indices of True values
        cov_indices = torch.cat([visible_indices * 2, visible_indices * 2 + 1]).sort()[0]
        filtered_var_hat = var_hat[batch_idx][cov_indices].unsqueeze(0)
        mean = means[batch_idx][cov_indices].unsqueeze(0)
        new_loss[i] = (torch.log(filtered_var_hat) + ((mean** 2) / filtered_var_hat)).mean()
    return new_loss.mean()


def beta_nll_loss(means: torch.Tensor, matrix: torch.Tensor, dim: int, gt: torch.Tensor) -> torch.Tensor:
    var_hat = matrix[:, :dim] ** 2

    visible_joints = (gt[:, 0, :, 2] >= -0.5).int()
    new_loss = torch.zeros(var_hat.size(0))
    for i, batch_idx in enumerate(range(var_hat.size(0))):
        visible_indices = visible_joints[batch_idx].nonzero(as_tuple=True)[0]  # Indices of True values
        cov_indices = torch.cat([visible_indices * 2, visible_indices * 2 + 1]).sort()[0]
        filtered_var_hat = var_hat[batch_idx][cov_indices].unsqueeze(0)
        mean = means[batch_idx][cov_indices].unsqueeze(0)
        new_loss[i] = ((torch.log(filtered_var_hat) + ((mean ** 2) / filtered_var_hat))*(torch.clone(filtered_var_hat).detach() ** 0.5)).mean()
    return new_loss.mean()


def faithful_loss(means: torch.Tensor, matrix: torch.Tensor, dim: int, gt: torch.Tensor) -> torch.Tensor:
    precision_hat = get_positive_definite_matrix(matrix, dim)

    visible_joints = (gt[:, 0, :, 2] >= -0.5).int()
    new_loss = torch.zeros(precision_hat.size(0))
    for batch_idx in range(precision_hat.size(0)):
        visible_indices = visible_joints[batch_idx].nonzero(as_tuple=True)[0]  # Indices of True values
        cov_indices = torch.cat([visible_indices * 2, visible_indices * 2 + 1]).sort()[0]
        filtered_precision_hat = precision_hat[batch_idx][cov_indices][:,cov_indices].unsqueeze(0)
        mean = means[batch_idx][cov_indices].unsqueeze(0)

        mse_loss = (mean ** 2).sum(dim=1)
        detached_ = mean.detach()
        nll_loss = -torch.logdet(filtered_precision_hat) + torch.matmul(
            torch.matmul(detached_.unsqueeze(1), filtered_precision_hat),
            detached_.unsqueeze(2)).squeeze()
        new_loss[batch_idx] = mse_loss + nll_loss

    return new_loss.mean()


def tic_loss(means: torch.Tensor, matrix: torch.Tensor, dim: int, pose_net: Union[ViTPose, Hourglass],
             pose_encodings: dict, use_hessian: bool, model_name: str, imgs: torch.Tensor, gt:torch.Tensor) -> torch.Tensor:
    
    psd_matrix = get_positive_definite_matrix(matrix, dim)
    covariance_hat = get_tic_covariance(
        pose_net, pose_encodings, matrix, psd_matrix, use_hessian, model_name, imgs)

    visible_joints = (gt[:, 0, :, 2] >= -0.5).int()
    new_loss = torch.zeros(covariance_hat.size(0))
    for i, batch_idx in enumerate(range(covariance_hat.size(0))):
        visible_indices = visible_joints[batch_idx].nonzero(as_tuple=True)[0]  # Indices of True values
        cov_indices = torch.cat([visible_indices * 2, visible_indices * 2 + 1]).sort()[0]
        filtered_covariance_hat = covariance_hat[batch_idx][cov_indices][:,cov_indices].unsqueeze(0)
        mean = means[batch_idx][cov_indices].unsqueeze(0)
        precision_hat = torch.linalg.inv(filtered_covariance_hat)

        new_loss[i] = -torch.logdet(precision_hat) + torch.matmul(
            torch.matmul(mean.unsqueeze(1), precision_hat),
            mean.unsqueeze(2)).squeeze()

    
    return new_loss.mean()