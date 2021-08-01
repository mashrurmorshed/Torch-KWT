from torch import nn, optim


def get_optimizer(net: nn.Module, opt_config: dict) -> optim.Optimizer:
    """Creates optimizer based on config.

    Args:
        net (nn.Module): Model instance.
        opt_config (dict): Dict containing optimizer settings.

    Raises:
        ValueError: Unsupported optimizer type.

    Returns:
        optim.Optimizer: Optimizer instance.
    """

    if opt_config["opt_type"] == "adamw":
        optimizer = optim.AdamW(net.parameters(), **opt_config["opt_kwargs"])
    else:
        raise ValueError(f'Unsupported optimizer {opt_config["opt_type"]}')

    return optimizer
