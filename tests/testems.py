import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import datetime
import numpy as np
from engine.ems import EventInterface

# ------------------- Test EventInterface ------------------- #
def test_eventinterface():
    class TestEventInterface(unittest.TestCase):
        def setUp(self):
            self.interface = EventInterface()
            self.interface.uid = 'test_uid'
            self.timestamp = datetime.datetime(2023, 10, 25, 15, 30)
            self.interface.now = self.timestamp
            self.symbol = self.interface.find_symbol_by_moneyness(self.timestamp, 'NIFTY', 0, 'CE', 0)

        def test_place_trade_success(self):
            success, price = self.interface.place_trade(
                timestamp=self.timestamp,
                action='BUY',
                qty=25,
                symbol=self.symbol,
                price=100.0
            )
            self.assertTrue(success)
            self.assertEqual(self.interface.trades[0]['symbol'], self.symbol)
            self.assertEqual(self.interface.trades[0]['qty_dir'], 25)

        def test_trade_position_clears(self):
            self.interface.place_trade(self.timestamp, 'BUY', 25, self.symbol, 100.0)
            self.interface.place_trade(self.timestamp, 'SELL', 25, self.symbol, 110.0)
            self.assertEqual(self.interface.positions, {})

        def test_invalid_trade_price_zero(self):
            success, price = self.interface.place_trade(self.timestamp, 'BUY', 25, self.symbol, 0.0)
            self.assertFalse(success)

        def test_invalid_trade_price_nan(self):
            success, price = self.interface.place_trade(self.timestamp, 'BUY', 25, self.symbol, np.nan)
            self.assertFalse(success)

        def test_get_mtm_empty(self):
            mtm = self.interface.get_mtm()
            self.assertEqual(mtm, 0)

        def test_process_event_bar_complete(self):
            event = {'timestamp': self.timestamp, 'bar_complete': True}
            self.interface.stop_time = datetime.time(15, 31)
            self.interface.process_event(event)
            self.assertEqual(self.interface.event, event)

        def test_get_active_trades_none(self):
            self.interface.trades = []
            self.interface.positions = {}
            active = self.interface.get_active_trades()
            self.assertEqual(active, [])

    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestEventInterface))


def main():
    print("Running EventInterface Tests...")
    test_eventinterface()

if __name__ == '__main__':
    main()
