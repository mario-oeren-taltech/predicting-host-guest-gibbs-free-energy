from pathlib import Path


def build_path(file_name: str) -> str:
    """
    Builds the absolute path for a given file used for tests.

    :param file_name: The name of the file.
    :return: The absolute path for the given file.
    """

    # Build the path for the folder with the files for tests.
    files_for_tests = Path(__file__).absolute().parent.joinpath('files_for_tests')

    # Build the absolute path to the given file.
    absolute_path = files_for_tests.joinpath(file_name)
    absolute_path.resolve(strict=True)

    return str(absolute_path)
