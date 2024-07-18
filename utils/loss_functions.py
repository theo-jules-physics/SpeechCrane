import torch
from itertools import chain


def fmap_loss(disc_out_dict: dict) -> torch.Tensor:
    """
    Computes the feature map loss between real and generated feature maps.

    This function calculates the mean absolute error between feature maps
    from the discriminator for real and generated data.

    Args:
        disc_out_dict (dict): Dictionary containing 'true_fmaps' and 'gen_fmaps', each being
        a list of lists of feature maps from the discriminators.

    Returns:
        torch.Tensor: Mean absolute error loss between the real and generated feature maps.
    """
    loss = 0.
    list_true_fmaps = list(chain(*disc_out_dict['true_fmaps']))
    list_gen_fmaps = list(chain(*disc_out_dict['gen_fmaps']))
    for dr, dg in zip(list_true_fmaps, list_gen_fmaps):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            max_len = min(rl.shape[2], gl.shape[2])
            rl = rl[:, :, :max_len]
            gl = gl[:, :, :max_len]
            loss = loss + torch.mean(torch.abs(rl - gl))
    return loss


def gloss(disc_out_dict: dict, loss_type: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Computes the generator loss for GANs using the specified loss type.

    Args:
        disc_out_dict (dict): Dictionary containing 'gen_outs', a list of lists of discriminators'
                              outputs for generated data.
        loss_type (str): Type of GAN loss ('lsgan' or 'hinge').

    Returns:
        tuple[torch.Tensor, list[torch.Tensor]]:
            - Total generator loss
            - List of individual losses for each discriminator output

    Raises:
        ValueError: If an unknown loss type is specified.
    """
    losses = 0.
    gen_loss = []
    list_gen_out = list(chain(*disc_out_dict['gen_outs']))
    for dg in list_gen_out:
        if loss_type == 'lsgan':
            loss = torch.mean((dg - 1) ** 2)
        elif loss_type == 'hinge':
            loss = -torch.mean(dg)
        else:
            raise ValueError(f"Unknown GAN loss: {loss_type}")
        gen_loss.append(loss)
        losses = losses + loss

    return losses, gen_loss


def dloss(disc_out_dict: dict, loss_type: str) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """
    Computes the discriminator loss for GANs using the specified loss type.

    Args:
        disc_out_dict (dict): Dictionary containing 'true_outs' and 'gen_outs',
        lists of lists of discriminators outputs for real and generated data.
        loss_type (str): Type of GAN loss ('lsgan' or 'hinge').

    Returns:
        tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
            - Total discriminator loss
            - List of losses for real data
            - List of losses for generated data

    Raises:
        ValueError: If an unknown loss type is specified.
    """
    loss = 0.0
    real_loss, gen_loss = [], []
    list_true_out = list(chain(*disc_out_dict['true_outs']))
    list_gen_out = list(chain(*disc_out_dict['gen_outs']))
    for dr, dg in zip(list_true_out, list_gen_out):
        if loss_type == 'lsgan':
            real_loss.append(torch.mean((dr - 1) ** 2))
            gen_loss.append(torch.mean(dg ** 2))
            loss = loss + real_loss[-1] + gen_loss[-1]
        elif loss_type == 'hinge':
            real_loss.append(-torch.mean(torch.min(dr - 1, torch.zeros_like(dr))))
            gen_loss.append(-torch.mean(torch.min(-dg - 1, torch.zeros_like(dg))))
            loss = loss + real_loss[-1] + gen_loss[-1]
        else:
            raise ValueError(f"Unknown GAN loss: {loss_type}")

    return loss, real_loss, gen_loss
