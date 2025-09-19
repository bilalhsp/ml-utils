

def match_dimensions(tensor, shape):
    """Return view of the input tensor to allow broadcasting with the shape."""
    tensor_shape = tensor.shape
    while len(tensor_shape) < len(shape):
        tensor_shape = tensor_shape + (1,) 
    return tensor.view(tensor_shape)