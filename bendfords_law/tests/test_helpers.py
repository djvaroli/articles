import numpy as np

from ..helpers import digit_occurrences, fib_sequence, get_digits, int_to_digits


def test_int_to_digits():
    """Tests that integers are correctly broken down into individual digits."""
    numbers_to_test = [100, 505, 8978]
    true_digits = [[1, 0, 0], [5, 0, 5], [8, 9, 7, 8]]
    for i_number, number in enumerate(numbers_to_test):
        digits = int_to_digits(number)
        assert (
            digits == true_digits[i_number]
        ), f"Obtained digit breakdown did not match expected breakdown. {digits} != {true_digits[i_number]}"


def test_get_first_digit():
    """Tests that the correct digit is extracted from an integer."""
    numbers_to_test = [100, 505, 8978]
    true_first_digits = [1, 5, 8]
    for i_number, number in enumerate(numbers_to_test):
        first_digit = get_digits(number, indices=[0])[0]
        assert (
            first_digit == true_first_digits[i_number]
        ), f"Digit mismatch {first_digit} != {true_first_digits[i_number]}."


def test_fib_sequence():
    """Test that Fibonacci sequences are correctly generated."""
    known_numbers = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    seq = fib_sequence(n_elements=10)
    assert (
        known_numbers == seq
    ).all(), (
        f"Known and Computed Fibonacci Sequence mismatch. {known_numbers} != {seq}."
    )


def test_digit_occurrences():
    """Tests that digit occurrences are correctly calculated."""
    numbers_to_test = np.arange(0, 10) * 100
    true_unique_digits = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9])
    true_counts = np.ones_like(true_unique_digits)
    digits, occurrences = digit_occurrences(numbers_to_test)
    assert (
        digits == true_unique_digits
    ).all(), f"Mismatch in obtained and expected unique digits. {digits} != {true_unique_digits}."
    assert (
        occurrences == true_counts
    ).all(), f"Mismatch in obtained and expected digit occurrences. {occurrences} != {true_counts}."
