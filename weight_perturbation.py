def update_weights(gradient, params, scale: float, lr: float):
    """
    Computes updated weights according to passed gradient and weight
    perturbation as learning rule

    Parameters
    ----------
    gradient : dict
        gradient estimate of the loss function
    params : dict
        current weights
    scale : float
        scaling factor of the perturbation (a.k.a. sigma)
    lr : float
        learning rate

    Returns
    -------
    dict
        updated weights

    """
    delta_W = {key: -lr / scale**2 * val for key, val in gradient.items()}
    return dictionary_add(delta_W, params)


def sample_perturbation(sampler, params):
    """
    Samples a perturbation dictionary with the same shape as params with the
    sampler's distribution

    Parameters
    ----------
    sampler : Distribution
        Sampling distribution for perturbation
    params : dict
        Weights and biases

    Returns
    -------
    h : dict
        Perturbation in the same shape as params
    """
    h = {}
    for key, val in params.items():
        h[key] = sampler.sample(sample_shape=val.shape)
    return h


def dictionary_add(dict1: dict, dict2: dict):
    """
    Computes the sum of two dictionaries per key.

    Parameters
    ----------
    dict1 : dict
    dict2 : dict

    Returns
    -------
    dict
        Sum of 2 dictionaries where sum[key] = dict1[key] + dict2[key]

    """
    return {key: val + dict2[key] for key, val in dict1.items()}


def dictionary_mult(dictionary, c):
    """
    Computes the multiplication of all dictionary items with a constant.

    Parameters
    ----------
    dictionary : dict
    c : Numeric

    Returns
    -------
    dict
        Original dictionary multiplied by c
    """
    return {key: val * c for key, val in dictionary.items()}


def compute_gradient(forward_pass, inputs, params, sampler, method="ffd"):
    """
    Computes the gradient according to weight perturbation with one of the
    following:
        - Forward Finite Difference (ffd)
        - Central Finite Difference (cfd)

    Parameters
    ----------
    forward_pass : Callable (inputs, params) -> loss
        Forward pass function
    inputs : Array
        Input data
    params : dict
        Network weights and biases
    sampler : Distribution
        Sampling distribution for perturbation
    method : str, optional
        Method for approximating gradient. Options are: {"ffd", "cfd"}.
        The default is "ffd".

    Raises
    ------
    ValueError
        If invalid method is given

    Returns
    -------
    gradient : dict
        Gradient estimation of the loss function wrt the weights as calculated
        by the specified method of weight perturbation.

    """
    h = sample_perturbation(sampler, params)
    if method == "ffd":
        # step 1: clean loss
        clean_loss = forward_pass(inputs, params)
        # step 2: perturbed loss
        perturbed_loss = forward_pass(inputs, dictionary_add(params, h))
        # step 3: gradient = (clean loss - perturbed loss) * h
        gradient = dictionary_mult(h, (perturbed_loss - clean_loss))
    elif method == "cfd":
        # step 1: perturbed loss
        perturbed_loss = forward_pass(inputs, dictionary_add(params, h))
        # step 2: reverse perturbed loss
        neg_perturbed_loss = forward_pass(
            inputs,
            dictionary_add(params, dictionary_mult(h, -1)),
        )
        # step 3: gradient = (clean loss - perturbed loss) * h
        gradient = dictionary_mult(h, (perturbed_loss - neg_perturbed_loss))
    else:
        raise ValueError('Invalid option given. Choose between: {"ffd", "cfd"}')
    return gradient


def compute_gradient_vector(forward_pass, inputs, params, sampler, method="ffd"):
    """
    Computes the gradient according to weight perturbation with one of the
    following:
        - Forward Finite Difference (ffd)
        - Central Finite Difference (cfd)

    Parameters
    ----------
    forward_pass : Callable (inputs, params) -> loss
        Forward pass function
    inputs : Array
        Input data
    params : dict
        Network weights and biases
    sampler : Distribution
        Sampling distribution for perturbation
    method : str, optional
        Method for approximating gradient. Options are: {"ffd", "cfd"}.
        The default is "ffd".

    Raises
    ------
    ValueError
        If invalid method is given

    Returns
    -------
    gradient : dict
        Gradient estimation of the loss function wrt the weights as calculated
        by the specified method of weight perturbation.

    """
    h = sample_perturbation(sampler, params)
    if method == "ffd":
        # step 1: clean loss
        clean_loss = forward_pass(inputs, params)
        # step 2: perturbed loss
        perturbed_loss = forward_pass(inputs, dictionary_add(params, h))
        # step 3: gradient = (clean loss - perturbed loss) * h
        gradient = dictionary_mult(h, (perturbed_loss - clean_loss))
    elif method == "cfd":
        # step 1: perturbed loss
        perturbed_loss = forward_pass(inputs, dictionary_add(params, h))
        # step 2: reverse perturbed loss
        neg_perturbed_loss = forward_pass(
            inputs,
            dictionary_add(params, dictionary_mult(h, -1)),
        )
        # step 3: gradient = (clean loss - perturbed loss) * h
        gradient = dictionary_mult(h, (perturbed_loss - neg_perturbed_loss))
    else:
        raise ValueError('Invalid option given. Choose between: {"ffd", "cfd"}')
    return gradient


def compute_snn_gradient(forward_pass, inputs, y, params, sampler, method="ffd"):
    """
    Computes the gradient according to weight perturbation with one of the
    following:
        - Forward Finite Difference (ffd)
        - Central Finite Difference (cfd)

    Parameters
    ----------
    forward_pass : Callable (inputs, params) -> loss
        Forward pass function
    inputs : Array
        Input data
    y : Array
        Target output
    params : dict
        Network weights and biases
    sampler : Distribution
        Sampling distribution for perturbation
    method : str, optional
        Method for approximating gradient. Options are: {"ffd", "cfd"}.
        The default is "ffd".

    Raises
    ------
    ValueError
        If invalid method is given

    Returns
    -------
    gradient : dict
        Gradient estimation of the loss function wrt the weights as calculated
        by the specified method of weight perturbation.

    """
    h = sample_perturbation(sampler, params)
    if method == "ffd":
        # step 1: clean loss
        clean_loss = forward_pass(inputs, y)
        # step 2: perturbed loss
        perturbed_loss = forward_pass(inputs, y, h)
        # step 3: gradient = (clean loss - perturbed loss) * h
        gradient = dictionary_mult(h, (perturbed_loss - clean_loss))
    elif method == "cfd":
        # step 1: perturbed loss
        perturbed_loss = forward_pass(inputs, y, h)
        # step 2: reverse perturbed loss
        neg_perturbed_loss = forward_pass(inputs, y, dictionary_mult(h, -1))
        # step 3: gradient = (clean loss - perturbed loss) * h
        gradient = dictionary_mult(h, (perturbed_loss - neg_perturbed_loss))
    else:
        raise ValueError('Invalid option given. Choose between: {"ffd", "cfd"}')
    return gradient
