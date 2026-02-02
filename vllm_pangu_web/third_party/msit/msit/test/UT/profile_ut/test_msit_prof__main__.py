import unittest
from unittest.mock import patch, MagicMock
import argparse

import sys
import types

# 导入待测模块
import components.profile.msit_prof.__main__ as msit_main

class TestCheckPositiveInteger(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(msit_main.check_positive_integer("5"), 5)
        self.assertEqual(msit_main.check_positive_integer(10), 10)

    def test_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            msit_main.check_positive_integer("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            msit_main.check_positive_integer("-1")

class TestCheckBatchsizeValid(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(msit_main.check_batchsize_valid(None))

    def test_valid(self):
        self.assertEqual(msit_main.check_batchsize_valid("3"), 3)

    def test_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            msit_main.check_batchsize_valid("0")

class TestCheckNonnegativeInteger(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(msit_main.check_nonnegative_integer("0"), 0)
        self.assertEqual(msit_main.check_nonnegative_integer(5), 5)

    def test_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            msit_main.check_nonnegative_integer("-1")

class TestCheckDeviceRangeValid(unittest.TestCase):
    def test_single_valid(self):
        self.assertEqual(msit_main.check_device_range_valid("0"), 0)
        self.assertEqual(msit_main.check_device_range_valid("255"), 255)

    def test_single_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            msit_main.check_device_range_valid("-1")
        with self.assertRaises(argparse.ArgumentTypeError):
            msit_main.check_device_range_valid("256")

    def test_list_valid(self):
        self.assertEqual(msit_main.check_device_range_valid("0,1,2"), [0, 1, 2])
        self.assertEqual(msit_main.check_device_range_valid("10,20,30"), [10, 20, 30])

    def test_list_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            msit_main.check_device_range_valid("0,256")
        with self.assertRaises(argparse.ArgumentTypeError):
            msit_main.check_device_range_valid("-1,2")

class TestGetArgs(unittest.TestCase):
    @patch("argparse.ArgumentParser.parse_args")
    def test_get_args(self, mock_parse_args):
        mock_args = MagicMock()
        mock_parse_args.return_value = mock_args
        args = msit_main.get_args()
        self.assertEqual(args, mock_args)

if __name__ == "__main__":
    unittest.main()