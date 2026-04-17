def map_probability_to_label(p_fake, low=0.35, high=0.65):
    """
    Map fake-probability to label using thresholds.
    """
    if p_fake < low:
        return "real"
    if p_fake > high:
        return "fake"
    return "suspicious"