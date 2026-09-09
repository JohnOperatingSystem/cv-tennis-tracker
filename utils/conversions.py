def _validate_reference_dimensions(reference_height_in_meters, reference_height_in_pixels):
    if reference_height_in_meters <= 0:
        raise ValueError("Reference height in meters must be greater than zero")
    if reference_height_in_pixels <= 0:
        raise ValueError("Reference height in pixels must be greater than zero")


def convert_pixel_distance_to_meters(
    pixel_distance, reference_height_in_meters, reference_height_in_pixels
):
    _validate_reference_dimensions(
        reference_height_in_meters, reference_height_in_pixels
    )
    return (pixel_distance / reference_height_in_pixels) * reference_height_in_meters


def convert_meters_to_pixel_distance(
    meter_distance, reference_height_in_meters, reference_height_in_pixels
):
    _validate_reference_dimensions(
        reference_height_in_meters, reference_height_in_pixels
    )
    return (meter_distance / reference_height_in_meters) * reference_height_in_pixels
