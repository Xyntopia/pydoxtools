from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import functools
import hashlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import pydantic

box_cols = ["x0", "y0", "x1", "y1"]
x0, y0, x1, y1 = 0, 1, 2, 3


@functools.lru_cache(maxsize=5)
def cached_pdf2image(file, dpi=100):
    import pdf2image
    # pdf2image.convert_from_bytes
    images = pdf2image.convert_from_path(file, dpi=dpi)  # first_page=None, last_page=None
    return images


def obj2float(text):
    """
    homogenous distribution of the hash value can be tested this way:
    pd.DataFrame([obj2float(i) for i in range(1000000)]).hist()
    """
    hashobj = hashlib.md5(str(text).encode('utf-8'))
    return abs(int(hashobj.hexdigest(), base=16)) / 2 ** 128


def color_from_string(text) -> np.array:
    """
    create a unique color for strings provided

    test uniform distribution of colors:

    cs = pd.DataFrame([color_from_string(i) for i in range(10000)]).hist()
    """
    a = obj2float(str(text) + "a")
    b = obj2float(str(text) + "b")
    c = obj2float(str(text) + "c")

    x = np.array([
        a, b, c
    ])
    return x / np.sqrt((x * x).sum())
    # return x/x.sum()


class LayerProps(pydantic.BaseModel):
    """
    A class to represent the properties of a layer in a plot.

    Attributes:
        linewidth (float): The width of the lines in the plot. Default is 0.2.
        alpha (float): The transparency of the lines in the plot. Default is 0.5.
        linestyle (str): The style of the lines in the plot. Default is "-".
        color (Optional[Union[Tuple[int, int, int], str]]): The color of the lines in the plot. Default is None.
        filled (bool): Whether the boxes in the plot should be filled. Default is True.
        box_numbers (bool): Whether to display box numbers in the plot. Default is False.
    """
    linewidth: float = 0.2
    alpha: float = 0.5
    linestyle: str = "-"
    color: typing.Optional[typing.Union[typing.Tuple[int, int, int], str]] = None
    filled: bool = True
    box_numbers = False


def plot_boxes(
        boxes: np.array,
        bbox: typing.List = None,
        layer_props: LayerProps = None,
        ax=None, groups: typing.List = None,
        dpi=120
):
    """
    Plots boxes with the given properties.

    Args:
        boxes (np.array): An array of boxes to plot.
        bbox (List, optional): The bounding box of the plot. Default is None.
        layer_props (LayerProps, optional): The properties of the layer in the plot. Default is None.
        ax (optional): The matplotlib axis to plot on. Default is None.
        groups (List, optional): The groups of boxes to plot. Default is None.
        dpi (int, optional): The resolution of the plot. Default is 120.

    Returns:
        ax: The matplotlib axis with the plotted boxes.
    """
    box_polygons = boxes[:, [x0, y0, x1, y0, x1, y1, x0, y1]].reshape(-1, 4, 2)

    if not layer_props:
        layer_props = LayerProps()
    elif layer_props.color is None:
        layer_props.color = "black"

    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)
        if bbox:
            ax.set_xlim(bbox[x0], bbox[x1])
            ax.set_ylim(bbox[y0], bbox[y1])
        plt.gca().set_aspect('equal', adjustable='box')  # set x & y scale the same

    for i, box in enumerate(box_polygons):
        if groups is not None:
            color = color_from_string(groups[i])
        elif layer_props.color == "random":
            color = color_from_string(i)
        else:
            color = layer_props.color

        if layer_props.filled:
            ax.fill(
                *box.T,
                linewidth=layer_props.linewidth, alpha=layer_props.alpha,
                linestyle=layer_props.linestyle, color=color
            )
        else:
            ax.plot(
                *box[[0, 1, 2, 3, 0]].T,
                linewidth=layer_props.linewidth, alpha=layer_props.alpha,
                linestyle=layer_props.linestyle, color=color
            )

        if layer_props.box_numbers:
            ax.text(*box[1], i, fontsize=5)

    return ax


def plot_box_layers(
        box_layers: typing.List,
        image: typing.Any = None,  # TODO: what image type?
        image_box: typing.List = None,
        bbox: typing.List = None,
        dpi=120
):
    """
    Plots several box layers on the same plot.

    Args:
        box_layers (List): A list of box layers to plot.
        image (Any, optional): The background image for the plot. This could be a rendering of a pdf file for example.
        image_box (List, optional): The bounding box of the image. Default is None.
        bbox (List, optional): The bounding box of the plot. Default is None.
        dpi (int, optional): The resolution of the plot. Default is 120.

    Returns:
        fig: The matplotlib figure with the plotted box layers.
    """
    fig, ax = plt.subplots(dpi=dpi)
    ax.set_xlim(bbox[x0], bbox[x1])
    ax.set_ylim(bbox[y0], bbox[y1])
    plt.gca().set_aspect('equal', adjustable='box')  # set x & y scale the same

    if image is not None:
        if image_box is not None:
            img_extent = [image_box[0], image_box[2], image_box[1], image_box[3]]
        else:
            img_extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
        ax.imshow(
            image,
            # aspect='auto',
            origin='upper',  # lower
            alpha=0.5,
            extent=img_extent
        )

    for num, (layer, layer_props) in enumerate(box_layers):
        if not layer_props.color:
            layer_props.color = color_from_string(num)
        plot_boxes(boxes=layer, bbox=bbox, layer_props=layer_props, ax=ax, groups=None)

    return fig
