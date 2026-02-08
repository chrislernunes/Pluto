import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import datetime
import numpy as np
from engine.datainterface import DataInterface

# ------------------- Test DataInterface ------------------- #
def test_datainterface():
    class TestDataInterface(unittest.TestCase):
        def setUp(self):
            self.interface = DataInterface()
            self.timestamp = datetime.datetime(2023, 11, 30, 15, 00)
            self.symbol = 'NIFTY23102618000CE'
            self.underlying = 'NIFTY'
            self.expiry_idx = 0

        def test_get_lot_size(self):
            self.assertEqual(self.interface.get_lot_size('NIFTY'), 25)

        def test_get_strike_diff(self):
            self.assertEqual(self.interface.get_strike_diff('NIFTY'), 50)

        def test_parse_strike_from_symbol(self):
            strike = self.interface.parse_strike_from_symbol(self.symbol)
            self.assertEqual(strike, 18000)

        def test_shift_strike_in_symbol(self):
            shifted = self.interface.shift_strike_in_symbol(self.symbol, 1)
            self.assertIn('18050CE', shifted)

        def test_replace_strike_in_symbol(self):
            replaced = self.interface.replace_strike_in_symbol(self.symbol, 18200)
            self.assertIn('18200CE', replaced)

        def test_get_dte_valid(self):
            dte = self.interface.get_dte(self.timestamp, self.symbol)
            self.assertTrue(isinstance(dte, int))
            self.assertGreaterEqual(dte, 0)

        def test_get_tick_none_symbol(self):
            tick = self.interface.get_tick(self.timestamp, None)
            self.assertIn('c', tick)
            self.assertTrue(np.isnan(tick['c']))

        def test_price_relations(self):
            self.atm_symbol = self.interface.find_symbol_by_moneyness(self.timestamp, self.underlying, self.expiry_idx, 'CE', 0)
            self.otm_symbol = self.interface.find_symbol_by_moneyness(self.timestamp, self.underlying, self.expiry_idx, 'CE', 2)
            self.itm_symbol = self.interface.find_symbol_by_moneyness(self.timestamp, self.underlying, self.expiry_idx, 'CE', -2)
            atm_price = self.interface.get_tick(self.timestamp, self.atm_symbol)['c']
            otm_price = self.interface.get_tick(self.timestamp, self.otm_symbol)['c']
            itm_price = self.interface.get_tick(self.timestamp, self.itm_symbol)['c']
            self.assertTrue(itm_price > atm_price > otm_price)
        
        def test_parse_invalid_symbol(self):
            with self.assertRaises(ValueError):
                self.interface.parse_strike_from_symbol("INVALIDSYMBOL")

        def test_option_delta_range(self):
            symbol = self.interface.find_symbol_by_moneyness(self.timestamp, self.underlying, 0, 'CE', 0)
            delta = self.interface.get_option_delta_iv(self.timestamp, symbol)
            print("Delta of " + str(symbol) + " is " + str(delta))
            self.assertTrue(0 <= delta <= 1) # ATM Delta should be around 0.5

        def test_ce_price_monotonicity(self):
            strikes = [-2, -1, 0, 1, 2]
            prices = [
                self.interface.get_tick(self.timestamp, self.interface.find_symbol_by_moneyness(self.timestamp, self.underlying, 0, 'CE', k))['c']
                for k in strikes
            ]
            self.assertTrue(all(x >= y for x, y in zip(prices, prices[1:])))

        def test_pe_price_monotonicity(self):
            strikes = [-2, -1, 0, 1, 2]
            prices = [
                self.interface.get_tick(self.timestamp, self.interface.find_symbol_by_moneyness(self.timestamp, self.underlying, 0, 'PE', k))['c']
                for k in strikes
            ]
            self.assertTrue(all(x >= y for x, y in zip(prices, prices[1:])))
    
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestDataInterface))


def main():
    print("Running DataInterface Tests...")
    test_datainterface()

if __name__ == '__main__':
    main()
