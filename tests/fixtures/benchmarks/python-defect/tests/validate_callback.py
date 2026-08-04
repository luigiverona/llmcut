import sys
import unittest

sys.path.insert(0, ".")

from app.callback import callback_timeout  # noqa: E402


class CallbackTest(unittest.TestCase):
    def test_timeout_is_seconds(self) -> None:
        self.assertEqual(callback_timeout(), 30)


if __name__ == "__main__":
    unittest.main()
