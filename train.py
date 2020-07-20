#!/usr/bin/env python3
import tensorflow as tf

from lib.pwcnet import build_network
from lib.tfrecord import get_reader


def main():
    net = build_network()
    net.fit(


if __name__ == '__main__':
    main()
