#!/usr/bin/env python3

import sys
from dataclasses import dataclass
import argcomplete
from simple_parsing import ArgumentParser, Serializable
from typing import Callable, List, Type, TypeVar

D = TypeVar("D")


def with_args(cls: Type[D], argv: List[str] = None, file=None):
    """
    Decorator for automatically adding parsed args from cli.
    """
    print('serial', isinstance(cls, Serializable))

    def decorator(main: Callable[[Type[D]], None]):
        def wrapper():
            parser = ArgumentParser()
            parser.add_arguments(cls, dest='opts')
            argcomplete.autocomplete(parser)
            args = parser.parse_args(sys.argv[1:] if argv is None else argv)
            return main(args.opts)
        return wrapper
    return decorator
