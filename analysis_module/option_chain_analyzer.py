
import logging
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

class OptionChainAnalyzer:
    """
    Analyzes option chain data to extract key metrics like PCR, Max Pain, and S/R levels.
    """
    
    def calculate_pcr(self, option_data: Dict) -> Optional[float]:
        """
        Calculate Put-Call Ratio (Total Put OI / Total Call OI).
        """
        try:
            total_call_oi = 0
            total_put_oi = 0
            
            records = option_data.get('records', {})
            data = records.get('data', [])
            
            for item in data:
                if 'CE' in item:
                    total_call_oi += item['CE'].get('openInterest', 0)
                if 'PE' in item:
                    total_put_oi += item['PE'].get('openInterest', 0)
            
            if total_call_oi == 0:
                logger.warning("OptionChainAnalyzer: Total Call OI is 0")
                return None
            
            pcr = total_put_oi / total_call_oi
            return round(pcr, 4)
            
        except Exception as e:
            logger.error(f"Error calculating PCR: {e}")
            return None

    def calculate_max_pain(self, option_data: Dict) -> Optional[float]:
        """
        Calculate Max Pain theory strike price.
        Max Pain is the strike where option writers lose the least amount of money.
        """
        try:
            records = option_data.get('records', {})
            data = records.get('data', [])
            
            strikes = []
            call_oi = {}
            put_oi = {}
            
            for item in data:
                strike = item['strikePrice']
                strikes.append(strike)
                call_oi[strike] = item.get('CE', {}).get('openInterest', 0)
                put_oi[strike] = item.get('PE', {}).get('openInterest', 0)
            
            if not strikes:
                return None
                
            # Sort strikes to ensure correct iteration
            strikes.sort()
            
            pain_values = {}
            
            for strike in strikes:
                current_pain = 0
                
                # Call Pain: (Spot - Strike) * OI for In-The-Money Calls (Spot > Strike)
                # If expiry is at 'strike', calls below 'strike' are ITM
                # Loss = (Strike_Expiry - Strike_Option) * OI
                # Here we assume expiry is at 'strike' to calculate pain AT that level
                
                # Pain if market expires at 'strike':
                
                # 1. Call Writers lose if Strike > Call_Strike
                # Loss = (Strike - Call_Strike) * Call_OI
                for s in strikes:
                    if s < strike:
                         current_pain += (strike - s) * call_oi.get(s, 0)
                         
                # 2. Put Writers lose if Strike < Put_Strike
                # Loss = (Put_Strike - Strike) * Put_OI
                for s in strikes:
                    if s > strike:
                        current_pain += (s - strike) * put_oi.get(s, 0)
                        
                pain_values[strike] = current_pain
                
            # Find strike with minimum pain
            max_pain_strike = min(pain_values, key=pain_values.get)
            return max_pain_strike
            
        except Exception as e:
            logger.error(f"Error calculating Max Pain: {e}")
            return None

    def calculate_atm_iv(self, option_data: Dict, spot_price: float) -> Optional[float]:
        """
        Calculate average Implied Volatility (IV) for ATM strikes.
        Returns average of Call IV and Put IV for the strike closest to spot price.
        """
        try:
            records = option_data.get('records', {})
            data = records.get('data', [])
            
            if not data or spot_price == 0:
                return None
                
            # Find ATM strike (closest to spot)
            atm_strike_data = min(data, key=lambda x: abs(x['strikePrice'] - spot_price))
            
            iv_sum = 0
            count = 0
            
            if 'CE' in atm_strike_data:
                iv = atm_strike_data['CE'].get('impliedVolatility', 0)
                if iv > 0:
                    iv_sum += iv
                    count += 1
                    
            if 'PE' in atm_strike_data:
                iv = atm_strike_data['PE'].get('impliedVolatility', 0)
                if iv > 0:
                    iv_sum += iv
                    count += 1
            
            if count == 0:
                return None
                
            return round(iv_sum / count, 2)
            
        except Exception as e:
            logger.error(f"Error calculating ATM IV: {e}")
            return None

    def analyze_oi_change(self, option_data: Dict, spot_price: float) -> Dict:
        """
        Analyze Change in Open Interest to determine sentiment.
        Calculates net OI change for strikes within 5% of spot.
        """
        try:
            records = option_data.get('records', {})
            data = records.get('data', [])
            
            if not data or spot_price <= 0:
                return {}
            
            # Filter for strikes within 5% range to focus on relevant activity
            relevant_data = [
                d for d in data 
                if abs(d['strikePrice'] - spot_price) < (spot_price * 0.05)
            ]
            
            total_call_change = 0
            total_put_change = 0
            
            for item in relevant_data:
                if 'CE' in item:
                    total_call_change += item['CE'].get('changeinOpenInterest', 0)
                if 'PE' in item:
                    total_put_change += item['PE'].get('changeinOpenInterest', 0)
            
            # Determine sentiment based on net flow
            # Call OI Increase > Put OI Increase -> Bearish (Resistance building)
            # Put OI Increase > Call OI Increase -> Bullish (Support building)
            
            sentiment = "NEUTRAL"
            difference = total_put_change - total_call_change
            
            # Significant difference threshold (e.g., 50k contracts)
            # Adjust based on instrument volume, but relative comparison is safer
            
            if total_call_change > 0 and total_put_change > 0:
                if total_put_change > (total_call_change * 1.5):
                    sentiment = "BULLISH"
                elif total_call_change > (total_put_change * 1.5):
                    sentiment = "BEARISH"
            
            # Handling unwinding scenarios (negative OI change)
            elif total_call_change < 0 and total_put_change < 0:
                 if abs(total_call_change) > abs(total_put_change) * 1.5:
                     sentiment = "BULLISH_UNWINDING" # Short covering
                 elif abs(total_put_change) > abs(total_call_change) * 1.5:
                     sentiment = "BEARISH_UNWINDING" # Long unwinding

            return {
                "total_call_change": total_call_change,
                "total_put_change": total_put_change,
                "sentiment": sentiment,
                "net_change_diff": difference
            }
            
        except Exception as e:
            logger.error(f"Error analyzing OI change: {e}")
            return {}

    def get_key_strikes(self, option_data: Dict) -> Dict:
        """
        Identify strikes with highest Open Interest for Support and Resistance.
        """
        try:
            records = option_data.get('records', {})
            data = records.get('data', [])
            
            call_oi_map = {}
            put_oi_map = {}
            
            for item in data:
                strike = item['strikePrice']
                if 'CE' in item:
                    call_oi_map[strike] = item['CE'].get('openInterest', 0)
                if 'PE' in item:
                    put_oi_map[strike] = item['PE'].get('openInterest', 0)
            
            # Find max OI strikes
            max_call_oi_strike = max(call_oi_map, key=call_oi_map.get) if call_oi_map else 0
            max_put_oi_strike = max(put_oi_map, key=put_oi_map.get) if put_oi_map else 0
            
            # Get top 3 levels
            top_calls = sorted(call_oi_map.items(), key=lambda x: x[1], reverse=True)[:3]
            top_puts = sorted(put_oi_map.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                "max_call_oi_strike": max_call_oi_strike,
                "max_put_oi_strike": max_put_oi_strike,
                "resistance_levels": [s for s, oi in top_calls],
                "support_levels": [s for s, oi in top_puts],
                "max_call_oi": call_oi_map.get(max_call_oi_strike, 0),
                "max_put_oi": put_oi_map.get(max_put_oi_strike, 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting key strikes: {e}")
            return {}

if __name__ == "__main__":
    # Integration test with Fetcher
    try:
        from data_module.option_chain_fetcher import OptionChainFetcher
        
        logging.basicConfig(level=logging.INFO)
        fetcher = OptionChainFetcher()
        analyzer = OptionChainAnalyzer()
        
        print("Fetching data...")
        data = fetcher.fetch_option_chain("NIFTY")
        
        if data:
            print("\n--- Analysis Results ---")
            
            pcr = analyzer.calculate_pcr(data)
            print(f"PCR: {pcr}")
            
            max_pain = analyzer.calculate_max_pain(data)
            print(f"Max Pain: {max_pain}")
            
            levels = analyzer.get_key_strikes(data)
            print(f"Max Call OI (Res): {levels['max_call_oi_strike']}")
            print(f"Max Put OI (Sup): {levels['max_put_oi_strike']}")
            print(f"Top 3 Resistances: {levels['resistance_levels']}")
            print(f"Top 3 Supports: {levels['support_levels']}")
            
        else:
            print("Failed to fetch data for analysis")
            
    except ImportError:
        print("Please run this from the project root to test imports")
