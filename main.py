import argparse

from src.base import ParseKwargs

default_hp = {
    "str": "a",
    "int": 1,
    "float": 1.5,
    "e": 1.e3
}


class ModelParser(ParseKwargs):
    CHOICES = default_hp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion with unfolding")
    parser.add_argument("--hp", nargs='*', action=ModelParser, help="Hyper parameters", default=default_hp)

    args = parser.parse_args()
    print(args.__dict__)
    args_dict = args.__dict__
