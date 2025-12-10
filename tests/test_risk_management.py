
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from datetime import datetime

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_module.trade_tracker import TradeTracker

class TestRiskManagement(unittest.TestCase):
    
    @patch("data_module.trade_tracker.firestore.Client")
    def test_atr_trailing_stop_update(self, MockFirestore):
        # Setup Mock DB
        mock_db = MockFirestore.return_value
        mock_collection = mock_db.collection.return_value
        
        # Create Tracker
        tracker = TradeTracker()
        tracker.db = mock_db # Ensure it uses our mock
        
        # 1. Simulate an Open Trade (LONG)
        # Entry: 100, SL: 95, ATR: 2.0
        # Initial Trail Gap: 2.0 * 1.5 = 3.0
        
        trade_id = "test_trade_1"
        trade_data = {
            "trade_id": trade_id,
            "instrument": "NIFTY",
            "signal_type": "BULLISH_BREAKOUT",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "atr": 2.0,
            "status": "OPEN",
            "timestamp": datetime.now()
        }
        
        # Mock Query Result
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = trade_data
        mock_doc.reference = MagicMock() # The document reference for updating
        
        # Mock the stream() method to return our trade
        mock_collection.where.return_value.stream.return_value = [mock_doc]
        
        # 2. Case A: Price moves up to 105
        # New Potential SL = Price (105) - (ATR(2)*1.5) = 105 - 3 = 102
        # 102 > 95 (Old SL) -> SHOULD UPDATE
        
        current_prices = {"NIFTY": 105.0}
        tracker.check_open_trades(current_prices)
        
        # Verify Update Call
        mock_doc.reference.update.assert_called_with({"stop_loss": 102.0})
        print("\n✅ Case A: Trailing SL updated correctly (95.0 -> 102.0)")
        
        # Reset Mock
        mock_doc.reference.update.reset_mock()
        
        # 3. Case B: Price drops to 98 (Pullback)
        # New Potential SL = 98 - 3 = 95
        # 95 <= 95 (Old SL is effectively 95 or 102 depending on persistence, logic relies on db value)
        # Let's say DB still has 95 for this test because we didn't actually write to DB.
        # But if we assume we are testing logic: 
        #   Potential SL (95) is NOT > Old SL (95). So NO Update.
        
        current_prices = {"NIFTY": 98.0}
        tracker.check_open_trades(current_prices)
        
        mock_doc.reference.update.assert_not_called()
        print("✅ Case B: Trailing SL did NOT loosen on pullback")

    @patch("data_module.trade_tracker.firestore.Client")
    def test_short_trailing_update(self, MockFirestore):
        # Setup Mock DB
        mock_db = MockFirestore.return_value
        tracker = TradeTracker()
        tracker.db = mock_db
        
        # SHORT Trade
        # Entry: 100, SL: 105, ATR: 2.0
        # Trail Gap: 3.0
        trade_data = {
            "trade_id": "test_short",
            "instrument": "NIFTY",
            "signal_type": "BEARISH_BREAKOUT",
            "entry_price": 100.0,
            "stop_loss": 105.0,
            "take_profit": 90.0,
            "atr": 2.0,
            "status": "OPEN",
            "timestamp": datetime.now()
        }
        
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = trade_data
        mock_db.collection.return_value.where.return_value.stream.return_value = [mock_doc]
        
        # Price moves down (Profit) to 95
        # New Potential SL = 95 + 3 = 98
        # 98 < 105 (Old SL) -> SHOULD UPDATE
        
        current_prices = {"NIFTY": 95.0}
        tracker.check_open_trades(current_prices)
        
        mock_doc.reference.update.assert_called_with({"stop_loss": 98.0})
        print("\n✅ SHORT: Trailing SL tightened correctly (105.0 -> 98.0)")

if __name__ == '__main__':
    unittest.main()
