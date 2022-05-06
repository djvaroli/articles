from collections import Counter
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def int_to_digits(integer: int) -> List[int]:
    """Given an integer, breaks it down into it's individual digits.

    Args:
        integer (int): number to break into digits

    Returns:
        List[int]: list of digits making up the integer
    """

    return [int(digit) for digit in str(integer)]


def get_digits(number: int, indices: List[int]) -> List[int]:
    """Given an integer, return a list of digits located at the specified indices in the input number.

    Args:
        number (int): number to get digit from.
        indices (List[int]): indices of digits to extract.

    Returns:
        List[int]: extracted digit
    """
    if not isinstance(number, (int, np.integer)):
        raise ValueError(f"Number must be an integer, but found type {type(number)}.")

    digits = int_to_digits(number)
    return [digits[index] for index in indices]


def bendford_dist(digits: np.ndarray) -> np.ndarray:
    """Returns the distribution of digit occurrence probabilities, as dictated by Bendford's Law.

    Args:
        digits (np.ndarray): an array of digits to compute occurrence distribution for.

    Returns:
        np.ndarray: distribution of digit occurrence.
    """

    def log_n(x: np.ndarray, n: int = 10) -> np.ndarray:
        """Computes log of base n using the base switching property.

        Args:
            x (np.ndarray): array of numbers
            n (int, optional): base. Defaults to 10.

        Returns:
            np.ndarray: log base 10 of specified values.
        """
        return np.log(x) / np.log(n)

    return log_n(1 + (1 / digits))


def bendford_dist_(order: int) -> np.ndarray:
    """Returns distribution of digit co-occurrences as predicted by Bendford's law

    Args:
        order (int): order of Bendford's law. Setting order to 1 is equivalent to the single digit Bendford's law.

    Raises:
        ValueError: if order is less than 1.

    Returns:
        np.ndarray: distribution of probabilities of occurrences of digit combinations
    """

    if order < 1:
        raise ValueError("Order cannot be less than 1.")

    digit_combinations = np.arange(np.power(10, order - 1), np.power(10, order))
    return np.log10(1 + 1 / digit_combinations)


# cache results for faster performance
@lru_cache(None)
def fib_number(n: int) -> int:
    """Returns number at specific location in Fibonacci sequence.

    Args:
        n (int): index of desired number in Fibonacci sequence.

    Raises:
        ValueError: in the event that n is less than 0

    Returns:
        int: nth number from Fibonacci sequence
    """

    if n < 0:
        raise ValueError("N cannot be less than 0.")

    if n < 2:
        return n
    return fib_number(n - 1) + fib_number(n - 2)


def fib_sequence(n_elements: int = 100) -> np.ndarray:
    """Returns the first n numbers in a Fibonacci sequence.

    Args:
        n_elements (int, optional): number of elements of Fibonacci sequence. Defaults to 100.

    Returns:
        np.ndarray: first n_elements digits of the Fibonacci sequence
    """
    sequence = np.zeros(n_elements)
    for i_element in range(n_elements):
        sequence[i_element] = fib_number(i_element)
    return sequence


def digit_occurrences(
    sequence: np.ndarray, digit_index: int = 0, exclude_zero: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a tuple consisting of an array of digits and an array of corresponding digit counts in the
    provided sequence.

    Args:
        sequence (np.ndarray): sequence of numbers to use as pool for counting digit occurrences.
        digit_index (int, optional): index of digits to count. Defaults to 0.
        exclude_zero (bool, optional): whether to include 0 as a valid digit or not.

    Returns:
        Tuple[np.ndarray, np.ndarray]: array of unique digits and array of counts of occurrences
    """
    # slice off first digit of integer
    first_digits_list: List[int] = [
        get_digits(number, [digit_index])[0] for number in sequence
    ]
    digit_counter = Counter(first_digits_list)
    if exclude_zero:
        digit_counter.pop(0, None)  # zero might not be a valid leading digit

    sorted_pairs = sorted(list(digit_counter.items()), key=lambda tup: tup[0])
    digit_occurrence_array = np.array(sorted_pairs)

    digits = digit_occurrence_array[:, 0]
    occurrences = digit_occurrence_array[:, 1]
    return digits, occurrences


# source https://gist.github.com/schlerp/5e4453b9a52deb5f600495d33eec407d
def _format_title(
    title: str, subtitle: Optional[str] = None, subtitle_font_size: int = 14
) -> str:
    """Formats title for Plotly plot

    Args:
        title (str): main title of plot.
        subtitle (str, optional): subtitle for plot. Defaults to None.
        subtitle_font_size (int, optional): font size. Defaults to 14.

    Returns:
        str: formatted title and, optionally, subtitle
    """
    title = f"<b>{title}</b>"
    if not subtitle:
        return title
    subtitle = f'<span style="font-size: {subtitle_font_size}px;">{subtitle}</span>'
    return f"{title}<br>{subtitle}"


def plot_oom_dist(
    sequence: np.ndarray,
    legend: str,
    title: str = "Order of Magnitude Distribution",
    template: str = "plotly_white",
    width: int = 500,
    height: int = 500,
    delta: float = 1e-5,
):
    """Creates a figure displaying the distribution of orders of magnitude of a sequence of numbers.

    Args:
        sequence (np.ndarray): sequence of numbers
        legend (str): label of associated with the data displayed in the plot
        title (str): the title of the plot
        template (str, optional): Plotly template. Defaults to "plotly_white".
        width (int, optional): width of graph. Defaults to 500.
        height (int, optional): height of graph. Defaults to 500.
        delta (float, optional): added to input to log function to avoid numerical overflows. Defaults to 1e-5.

    Returns:
        go.Figure: figure with distribution
    """
    oom = np.floor(np.log10(sequence + delta))
    figure = go.Figure()
    figure.add_trace(go.Histogram(x=oom))
    figure.update_layout(
        title=_format_title(title, legend),
        width=width,
        height=height,
        template=template,
        xaxis_title="Order of Magnitude",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.85,
        ),
    )
    return figure


def _create_comparison_figure(
    digits: np.ndarray,
    occurrences: np.ndarray,
    trace_label: str,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    color: str = px.colors.qualitative.Dark2[3],
    template: str = "plotly_white",
    width: int = 500,
    height: int = 500,
):
    """Create a figure plotting the distribution of first digits predicted by Bendford's law for the specified digits
    and the observed distribution of first digit occurrences.

    Args:
        digits (np.ndarray): unique digits, to serve as data for x-axis.
        occurrences (np.ndarray): number of times each digit occurred in dataset, to serve as data for y-axis.
        trace_label (str): the name/label of the trace.
        title (str, optional): title for the generated plot.
        subtitle(str, optional): subtitle for the generated plot.
        color (str, optional): trace color. Defaults to px.colors.qualitative.Dark2[3].
        template (str, optional): plotly figure template. Defaults to "plotly_white".
        width (int, optional): width of figure
        height (int, optional): height of figure

    Returns:
        plotly graphing objects figure
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=digits,
            y=bendford_dist(digits),
            name="Bendford's Law.",
            line=dict(color=color),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=digits,
            y=occurrences / occurrences.sum(),
            name=trace_label,
            line=dict(color="MediumPurple"),
        )
    )
    fig.update_layout(
        title=_format_title(title, subtitle) if title else None,
        xaxis_title="Digit",
        yaxis_title="Probability",
        template=template,
        width=width,
        height=height,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.85,
        ),
    )

    return fig


def bendford_comparison_figure(
    sequence: np.ndarray,
    legend: str,
    *,
    title: str = "Digit Occurrence Distribution.",
    display_count: bool = True,
    color: str = px.colors.qualitative.Dark2[3],
    template: str = "plotly_white",
    width: int = 900,
    height: int = 700,
):
    """Create a figure plotting the distribution of first digits predicted by Bendford's law for the specified digits
    and the observed distribution of first digit occurrences.

    Args:
        sequence (np.ndarray): array of integers to be used as source of numbers to compute the digit occurrence distribution,
            that is to be compared against the distribution according to Bendford's law.
        legend (str): legend associated with the dataset used to generate the plot.
        title (str, optional): title for the generated plot.
        display_count(bool, optional): whether to display the total count of digits.
        color (str, optional): trace color. Defaults to px.colors.qualitative.Dark2[3].
        template (str, optional): plotly figure template. Defaults to "plotly_white".
        width (int, optional): width of figure
        height (int, optional): height of figure

    Returns:
        plotly graphing objects figure
    """
    digits, occurrences = digit_occurrences(sequence)
    subtitle: str = ""
    if display_count:
        subtitle = f"N = {occurrences.sum()}."

    figure = _create_comparison_figure(
        digits,
        occurrences,
        legend,
        title=title,
        subtitle=subtitle,
        color=color,
        template=template,
        width=width,
        height=height,
    )
    return figure
