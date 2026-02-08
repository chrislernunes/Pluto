import unittest
import datetime
from unittest.mock import MagicMock

from strategies.tbs import TBS

class TestTBSStrategy(unittest.TestCase):
    def setUp(self):
        self.tbs = TBS()
        uid = "tbs_0_x0_0_1_NIFTY_P_20_0_0.25_0.5_0_True_True"
        self.tbs.set_params_from_uid(uid)
        self.tbs.now = datetime.datetime(2023, 10, 25, 9, 20)
        self.tbs.get_lot_size = MagicMock(return_value=50)
        self.tbs.get_tick = MagicMock(return_value={'c': 100})
        self.tbs.find_symbol_by_premium = MagicMock(return_value='NIFTY23102618000CE')
        self.tbs.place_spread_trade = MagicMock(return_value=(True, 100, 5))
        self.tbs.on_new_day()

    def test_time_based_entry(self):
        self.tbs.on_bar_complete()
        self.assertEqual(self.tbs.position_ce, 1)
        self.assertEqual(self.tbs.position_pe, 1)

    def test_stop_loss_exit(self):
        self.tbs.position_ce = 1
        self.tbs.symbol_ce = 'NIFTY23102618000CE'
        self.tbs.symbol_ce_hedge = 'NIFTY23102618100CE'
        self.tbs.entry_price_ce = 100
        self.tbs.sl_price_ce = 110
        self.tbs.get_tick = MagicMock(return_value={'c': 111})
        self.tbs.place_spread_trade = MagicMock(return_value=(True, 111, 5))
        self.tbs.now = datetime.datetime(2023, 10, 25, 9, 21)
        self.tbs.on_bar_complete()
        self.assertEqual(self.tbs.position_ce, -1)

    def test_target_exit(self):
        self.tbs.position_pe = 1
        self.tbs.symbol_pe = 'NIFTY23102618000PE'
        self.tbs.symbol_pe_hedge = 'NIFTY23102618100PE'
        self.tbs.entry_price_pe = 100
        self.tbs.tgt_price_pe = 80
        self.tbs.Target = True
        self.tbs.get_tick = MagicMock(return_value={'c': 75})
        self.tbs.place_spread_trade = MagicMock(return_value=(True, 75, 5))
        self.tbs.now = datetime.datetime(2023, 10, 25, 9, 21)
        self.tbs.on_bar_complete()
        self.assertEqual(self.tbs.position_pe, -1)

    def test_no_reentry_after_exit(self):
        self.test_target_exit()  # Forces PE exit
        self.tbs.now = datetime.datetime(2023, 10, 25, 9, 22)
        self.tbs.on_bar_complete()
        self.assertNotEqual(self.tbs.position_pe, 0)

    def test_end_of_day_closure(self):
        self.tbs.position_ce = 1
        self.tbs.symbol_ce = 'NIFTY23102618000CE'
        self.tbs.symbol_ce_hedge = 'NIFTY23102618100CE'
        self.tbs.sl_price_ce = 150
        self.tbs.tgt_price_ce = 50
        self.tbs.get_tick = MagicMock(return_value={'c': 90})
        self.tbs.place_spread_trade = MagicMock(return_value=(True, 90, 5))
        self.tbs.now = datetime.datetime.combine(datetime.date.today(), self.tbs.stop_time)
        self.tbs.on_bar_complete()
        self.assertEqual(self.tbs.position_ce, -1)

if __name__ == '__main__':
    unittest.main()
