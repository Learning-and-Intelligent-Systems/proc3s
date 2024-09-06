import collections

import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from easydict import EasyDict
from PIL import Image
from tqdm import tqdm

STANDARD_COLORS = ["White"]
FLAGS = {
    "prompt_engineering": True,
    "this_is": True,
    "temperature": 100.0,
    "use_softmax": False,
}
FLAGS = EasyDict(FLAGS)


# @markdown Non-maximum suppression (NMS).
def nms(dets, scores, thresh, max_dets=1000):
    """Non-maximum suppression.

    Args:
      dets: [N, 4]
      scores: [N,]
      thresh: iou threshold. Float
      max_dets: int.
    """
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0 and len(keep) < max_dets:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        overlap = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-12)

        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]
    return keep


def draw_bounding_box_on_image(
    image,
    ymin,
    xmin,
    ymax,
    xmax,
    color="red",
    thickness=4,
    display_str_list=(),
    use_normalized_coordinates=True,
):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input "color".
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (
            xmin * im_width,
            xmax * im_width,
            ymin * im_height,
            ymax * im_height,
        )
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_left = min(5, left)
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )
        text_bottom -= text_height - 2 * margin


def draw_bounding_box_on_image_array(
    image,
    ymin,
    xmin,
    ymax,
    xmax,
    color="red",
    thickness=4,
    display_str_list=(),
    use_normalized_coordinates=True,
):
    """Adds a bounding box to an image (numpy array).

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Args:
      image: a numpy array with shape [height, width, 3].
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    draw_bounding_box_on_image(
        image_pil,
        ymin,
        xmin,
        ymax,
        xmax,
        color,
        thickness,
        display_str_list,
        use_normalized_coordinates,
    )
    np.copyto(image, np.array(image_pil))


def draw_mask_on_image_array(image, mask, color="red", alpha=0.4):
    """Draws mask on an image.

    Args:
      image: uint8 numpy array with shape (img_height, img_height, 3)
      mask: a uint8 numpy array of shape (img_height, img_height) with
        values between either 0 or 1.
      color: color to draw the keypoints with. Default is red.
      alpha: transparency value between 0 and 1. (default: 0.4)

    Raises:
      ValueError: On incorrect data type for image or masks.
    """
    if image.dtype != np.uint8:
        raise ValueError("`image` not of type np.uint8")
    if mask.dtype != np.uint8:
        raise ValueError("`mask` not of type np.uint8")
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError("`mask` elements should be in [0, 1]")
    if image.shape[:2] != mask.shape:
        raise ValueError(
            "The image has spatial dimensions %s but the mask has "
            "dimensions %s" % (image.shape[:2], mask.shape)
        )
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(
        list(rgb), [1, 1, 3]
    )
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert("RGBA")
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert("L")
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert("RGB")))


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=0.5,
    agnostic_mode=False,
    line_thickness=1,
    groundtruth_box_visualization_color="black",
    skip_scores=False,
    skip_labels=False,
    mask_alpha=0.4,
    plot_color=None,
):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      instance_masks: a numpy array of shape [N, image_height, image_width] with
        values ranging between 0 and 1, can be None.
      instance_boundaries: a numpy array of shape [N, image_height, image_width]
        with values ranging between 0 and 1, can be None.
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      min_score_thresh: minimum score threshold for a box to be visualized
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
      line_thickness: integer (default: 4) controlling line width of the boxes.
      groundtruth_box_visualization_color: box color for visualizing groundtruth
        boxes
      skip_scores: whether to skip score when drawing a single detection
      skip_labels: whether to skip label when drawing a single detection

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_score_map = {}
    box_to_instance_boundaries_map = {}

    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ""
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in list(category_index.keys()):
                            class_name = category_index[classes[i]]["name"]
                        else:
                            class_name = "N/A"
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = "{}%".format(int(100 * scores[i]))
                    else:
                        float_score = ("%.2f" % scores[i]).lstrip("0")
                        display_str = "{}: {}".format(display_str, float_score)
                    box_to_score_map[box] = int(100 * scores[i])

                box_to_display_str_map[box].append(display_str)
                if plot_color is not None:
                    box_to_color_map[box] = plot_color
                elif agnostic_mode:
                    box_to_color_map[box] = "DarkOrange"
                else:
                    box_to_color_map[box] = STANDARD_COLORS[
                        classes[i] % len(STANDARD_COLORS)
                    ]

    # Handle the case when box_to_score_map is empty.
    if box_to_score_map:
        box_color_iter = sorted(
            box_to_color_map.items(), key=lambda kv: box_to_score_map[kv[0]]
        )
    else:
        box_color_iter = box_to_color_map.items()

    # Draw all boxes onto image.
    for box, color in box_color_iter:
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
            draw_mask_on_image_array(
                image, box_to_instance_masks_map[box], color=color, alpha=mask_alpha
            )
        if instance_boundaries is not None:
            draw_mask_on_image_array(
                image, box_to_instance_boundaries_map[box], color="red", alpha=1.0
            )
        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates,
        )

    return image


def paste_instance_masks(masks, detected_boxes, image_height, image_width):
    """Paste instance masks to generate the image segmentation results.

    Args:
      masks: a numpy array of shape [N, mask_height, mask_width] representing the
        instance masks w.r.t. the `detected_boxes`.
      detected_boxes: a numpy array of shape [N, 4] representing the reference
        bounding boxes.
      image_height: an integer representing the height of the image.
      image_width: an integer representing the width of the image.

    Returns:
      segms: a numpy array of shape [N, image_height, image_width] representing
        the instance masks *pasted* on the image canvas.
    """

    def expand_boxes(boxes, scale):
        """Expands an array of boxes by a given scale."""
        # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L227  # pylint: disable=line-too-long
        # The `boxes` in the reference implementation is in [x1, y1, x2, y2] form,
        # whereas `boxes` here is in [x1, y1, w, h] form
        w_half = boxes[:, 2] * 0.5
        h_half = boxes[:, 3] * 0.5
        x_c = boxes[:, 0] + w_half
        y_c = boxes[:, 1] + h_half

        w_half *= scale
        h_half *= scale

        boxes_exp = np.zeros(boxes.shape)
        boxes_exp[:, 0] = x_c - w_half
        boxes_exp[:, 2] = x_c + w_half
        boxes_exp[:, 1] = y_c - h_half
        boxes_exp[:, 3] = y_c + h_half

        return boxes_exp

    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/test.py#L812  # pylint: disable=line-too-long
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    _, mask_height, mask_width = masks.shape
    scale = max((mask_width + 2.0) / mask_width, (mask_height + 2.0) / mask_height)

    ref_boxes = expand_boxes(detected_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
    segms = []
    for mask_ind, mask in enumerate(masks):
        im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        # Process mask inside bounding boxes.
        padded_mask[1:-1, 1:-1] = mask[:, :]

        ref_box = ref_boxes[mask_ind, :]
        w = ref_box[2] - ref_box[0] + 1
        h = ref_box[3] - ref_box[1] + 1
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)

        mask = cv2.resize(padded_mask, (w, h))
        mask = np.array(mask > 0.5, dtype=np.uint8)

        x_0 = min(max(ref_box[0], 0), image_width)
        x_1 = min(max(ref_box[2] + 1, 0), image_width)
        y_0 = min(max(ref_box[1], 0), image_height)
        y_1 = min(max(ref_box[3] + 1, 0), image_height)

        im_mask[y_0:y_1, x_0:x_1] = mask[
            (y_0 - ref_box[1]) : (y_1 - ref_box[1]),
            (x_0 - ref_box[0]) : (x_1 - ref_box[0]),
        ]
        segms.append(im_mask)

    segms = np.array(segms)
    assert masks.shape[0] == segms.shape[0]
    return segms


# @markdown Plot instance masks.
def plot_mask(color, alpha, original_image, mask):
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(original_image)

    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(
        list(rgb), [1, 1, 3]
    )
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert("RGBA")
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert("L")
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    img_w_mask = np.array(pil_image.convert("RGB"))
    return img_w_mask


def display_image(path_or_array, size=(10, 10)):
    if isinstance(path_or_array, str):
        image = np.asarray(Image.open(open(image_path, "rb")).convert("RGB"))
    else:
        image = path_or_array

    plt.figure(figsize=size)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
