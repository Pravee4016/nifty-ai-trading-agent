"""
System Validation Script
Comprehensive pre-production health check for the trading system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemValidator:
    """Validates all system components."""
    
    def __init__(self):
        self.results = []
        self.errors = []
        self.warnings = []
    
    def test(self, name, func):
        """Run a test and record result."""
        try:
            logger.info(f"Testing: {name}...")
            func()
            self.results.append((name, "✅ PASS", None))
            logger.info(f"  ✅ PASS: {name}")
            return True
        except Exception as e:
            error_msg = str(e)
            self.results.append((name, "❌ FAIL", error_msg))
            self.errors.append((name, error_msg))
            logger.error(f"  ❌ FAIL: {name} - {error_msg}")
            return False
    
    def warn(self, name, message):
        """Record a warning."""
        self.warnings.append((name, message))
        logger.warning(f"  ⚠️ WARNING: {name} - {message}")
    
    def run_all_tests(self):
        """Run all validation tests."""
        logger.info("="*80)
        logger.info("🔍 SYSTEM VALIDATION - PRE-PRODUCTION HEALTH CHECK")
        logger.info("="*80)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        # Test 1: Core Imports
        self.test("Core Python Imports", self.test_core_imports)
        
        # Test 2: Configuration
        self.test("Configuration Loading", self.test_configuration)
        
        # Test 3: Technical Analyzer
        self.test("Technical Analyzer", self.test_technical_analyzer)
        
        # Test 4: MACD Calculation
        self.test("MACD Calculation", self.test_macd)
        
        # Test 5: EMA Crossover
        self.test("EMA Crossover Detection", self.test_ema_crossover)
        
        # Test 6: Combo Signals
        self.test("Combo Signal Evaluator", self.test_combo_signals)
        
        # Test 7: Signal Pipeline
        self.test("Signal Pipeline Integration", self.test_signal_pipeline)
        
        # Test 8: Market State Engine
        self.test("Market State Engine", self.test_market_state)
        
        # Test 9: Data Dependencies
        self.test("Required Libraries", self.test_dependencies)
        
        # Test 10: Feature Flags
        self.test("Feature Flags", self.test_feature_flags)
        
        # Test 11: File System
        self.test("File System Access", self.test_file_system)
        
        # Generate Report
        return self.generate_report()
    
    def test_core_imports(self):
        """Test core Python imports."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        import logging
        assert pd.__version__, "Pandas not available"
        assert np.__version__, "NumPy not available"
    
    def test_configuration(self):
        """Test configuration loading."""
        from config import settings
        
        # Check critical settings exist
        assert hasattr(settings, 'USE_COMBO_SIGNALS'), "USE_COMBO_SIGNALS not in config"
        assert hasattr(settings, 'EMA_CROSSOVER_FAST'), "EMA_CROSSOVER_FAST not in config"
        assert hasattr(settings, 'EMA_CROSSOVER_SLOW'), "EMA_CROSSOVER_SLOW not in config"
        assert hasattr(settings, 'MACD_FAST'), "MACD config missing"
        assert hasattr(settings, 'MACD_SLOW'), "MACD config missing"
        assert hasattr(settings, 'MACD_SIGNAL'), "MACD config missing"
        
        # Verify values
        assert settings.EMA_CROSSOVER_FAST == 5, "EMA_CROSSOVER_FAST should be 5"
        assert settings.EMA_CROSSOVER_SLOW == 15, "EMA_CROSSOVER_SLOW should be 15"
        assert settings.USE_COMBO_SIGNALS == True, "USE_COMBO_SIGNALS should be True"
    
    def test_technical_analyzer(self):
        """Test Technical Analyzer initialization and basic methods."""
        from analysis_module.technical import TechnicalAnalyzer
        import pandas as pd
        import numpy as np
        
        analyzer = TechnicalAnalyzer("TEST")
        
        # Create sample data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='5T')
        df = pd.DataFrame({
            'open': np.random.uniform(25000, 26000, 100),
            'high': np.random.uniform(25100, 26100, 100),
            'low': np.random.uniform(24900, 25900, 100),
            'close': np.random.uniform(25000, 26000, 100),
            'volume': np.random.uniform(10000, 50000, 100)
        }, index=dates)
        
        # Test basic calculations
        rsi = analyzer._calculate_rsi(df)
        assert isinstance(rsi, (int, float)), "RSI calculation failed"
        assert 0 <= rsi <= 100, "RSI out of range"
        
        atr = analyzer._calculate_atr(df)
        assert isinstance(atr, (int, float)), "ATR calculation failed"
        assert atr >= 0, "ATR should be positive"
    
    def test_macd(self):
        """Test MACD calculation."""
        from analysis_module.technical import TechnicalAnalyzer
        import pandas as pd
        import numpy as np
        
        analyzer = TechnicalAnalyzer("TEST")
        
        # Create sample data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='5T')
        df = pd.DataFrame({
            'close': np.random.uniform(25000, 26000, 100)
        }, index=dates)
        
        macd_data = analyzer._calculate_macd(df)
        
        assert 'macd_line' in macd_data, "MACD line missing"
        assert 'signal_line' in macd_data, "Signal line missing"
        assert 'histogram' in macd_data, "Histogram missing"
        assert 'crossover' in macd_data, "Crossover missing"
        assert macd_data['crossover'] in ['BULLISH', 'BEARISH', 'NONE'], "Invalid crossover value"
    
    def test_ema_crossover(self):
        """Test EMA crossover detection."""
        from analysis_module.technical import TechnicalAnalyzer
        import pandas as pd
        import numpy as np
        
        analyzer = TechnicalAnalyzer("TEST")
        
        dates = pd.date_range(start='2025-01-01', periods=100, freq='5T')
        df = pd.DataFrame({
            'close': np.random.uniform(25000, 26000, 100)
        }, index=dates)
        
        ema_data = analyzer.detect_ema_crossover(df)
        
        assert 'bias' in ema_data, "Bias missing"
        assert 'confidence' in ema_data, "Confidence missing"
        assert ema_data['bias'] in ['BULLISH', 'BEARISH', 'NEUTRAL'], "Invalid bias"
        assert 0 <= ema_data['confidence'] <= 1, "Confidence out of range"
    
    def test_combo_signals(self):
        """Test combo signal evaluator."""
        from analysis_module.combo_signals import MACDRSIBBCombo
        from analysis_module.technical import TechnicalAnalyzer
        import pandas as pd
        import numpy as np
        
        combo = MACDRSIBBCombo()
        analyzer = TechnicalAnalyzer("TEST")
        
        dates = pd.date_range(start='2025-01-01', periods=100, freq='5T')
        df = pd.DataFrame({
            'open': np.random.uniform(25000, 26000, 100),
            'high': np.random.uniform(25100, 26100, 100),
            'low': np.random.uniform(24900, 25900, 100),
            'close': np.random.uniform(25000, 26000, 100),
            'volume': np.random.uniform(10000, 50000, 100)
        }, index=dates)
        
        # Build technical context
        macd_data = analyzer._calculate_macd(df)
        bb_data = analyzer._calculate_bollinger_bands(df)
        rsi = analyzer._calculate_rsi(df)
        
        technical_context = {
            "macd": macd_data,
            "rsi_5": rsi,
            "bb_upper": bb_data['upper'].iloc[-1],
            "bb_lower": bb_data['lower'].iloc[-1]
        }
        
        result = combo.evaluate_signal(df, "BULLISH", technical_context)
        
        assert 'strength' in result, "Strength missing"
        assert 'score' in result, "Score missing"
        assert result['strength'] in ['STRONG', 'MEDIUM', 'WEAK', 'INVALID'], "Invalid strength"
        assert 0 <= result['score'] <= 3, "Score out of range"
    
    def test_signal_pipeline(self):
        """Test signal pipeline integration."""
        from analysis_module.signal_pipeline import SignalPipeline
        
        pipeline = SignalPipeline()
        
        # Check combo evaluator initialized
        assert hasattr(pipeline, 'combo_evaluator'), "Combo evaluator not initialized"
        assert pipeline.combo_evaluator is not None, "Combo evaluator is None"
        
        # Check market state engine
        assert hasattr(pipeline, 'state_engine'), "State engine not initialized"
        assert pipeline.state_engine is not None, "State engine is None"
    
    def test_market_state(self):
        """Test market state engine."""
        from analysis_module.market_state_engine import MarketStateEngine, MarketState
        import pandas as pd
        import numpy as np
        
        engine = MarketStateEngine()
        
        dates = pd.date_range(start='2025-01-01', periods=100, freq='5T')
        df = pd.DataFrame({
            'open': np.random.uniform(25000, 26000, 100),
            'high': np.random.uniform(25100, 26100, 100),
            'low': np.random.uniform(24900, 25900, 100),
            'close': np.random.uniform(25000, 26000, 100),
            'volume': np.random.uniform(10000, 50000, 100)
        }, index=dates)
        
        state_info = engine.evaluate_state(df)
        
        assert 'state' in state_info, "State missing"
        assert 'confidence' in state_info, "Confidence missing"
        assert isinstance(state_info['state'], MarketState), "Invalid state type"
        assert 0 <= state_info['confidence'] <= 1, "Confidence out of range"
    
    def test_dependencies(self):
        """Test required libraries."""
        try:
            import pandas
            import numpy
            import yfinance
            import pytz
        except ImportError as e:
            raise Exception(f"Missing dependency: {e}")
    
    def test_feature_flags(self):
        """Test feature flags."""
        from config import settings
        
        flags = {
            'USE_COMBO_SIGNALS': settings.USE_COMBO_SIGNALS,
            'USE_ML_FILTERING': settings.USE_ML_FILTERING,
            'USE_EXPERT_ENHANCEMENTS': getattr(settings, 'USE_EXPERT_ENHANCEMENTS', None),
        }
        
        logger.info(f"  Feature Flags: {flags}")
        
        # Combo should be enabled
        if not flags['USE_COMBO_SIGNALS']:
            self.warn("Feature Flags", "USE_COMBO_SIGNALS is disabled")
    
    def test_file_system(self):
        """Test file system access."""
        import os
        
        # Check key directories exist
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        required_dirs = [
            'analysis_module',
            'config',
            'scripts'
        ]
        
        for dir_name in required_dirs:
            dir_path = os.path.join(base_dir, dir_name)
            assert os.path.exists(dir_path), f"Directory missing: {dir_name}"
        
        # Check key files exist
        required_files = [
            'analysis_module/technical.py',
            'analysis_module/combo_signals.py',
            'analysis_module/signal_pipeline.py',
            'config/settings.py'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(base_dir, file_name)
            assert os.path.exists(file_path), f"File missing: {file_name}"
    
    def generate_report(self):
        """Generate validation report."""
        logger.info("")
        logger.info("="*80)
        logger.info("📊 VALIDATION REPORT")
        logger.info("="*80)
        
        total_tests = len(self.results)
        passed = sum(1 for _, status, _ in self.results if status == "✅ PASS")
        failed = sum(1 for _, status, _ in self.results if status == "❌ FAIL")
        
        logger.info(f"\nTotal Tests: {total_tests}")
        logger.info(f"Passed: {passed} ✅")
        logger.info(f"Failed: {failed} ❌")
        logger.info(f"Warnings: {len(self.warnings)} ⚠️")
        
        logger.info("\n" + "="*80)
        logger.info("Detailed Results:")
        logger.info("="*80)
        
        for name, status, error in self.results:
            if error:
                logger.info(f"{status} {name}: {error}")
            else:
                logger.info(f"{status} {name}")
        
        if failed == 0 and len(self.warnings) == 0:
            logger.info("\n" + "="*80)
            logger.info("✅ ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION")
            logger.info("="*80)
            return True
        elif failed == 0:
            logger.info("\n" + "="*80)
            logger.info("⚠️ ALL TESTS PASSED WITH WARNINGS - REVIEW WARNINGS")
            logger.info("="*80)
            
            logger.info("\nWarnings:")
            for name, message in self.warnings:
                logger.info(f"  ⚠️ {name}: {message}")
            return True  # Still return True if only warnings
        else:
            logger.info("\n" + "="*80)
            logger.info("❌ VALIDATION FAILED - FIX ERRORS BEFORE PRODUCTION")
            logger.info("="*80)
            
            logger.info("\nErrors:")
            for name, error in self.errors:
                logger.info(f"  ❌ {name}: {error}")
            
            if self.warnings:
                logger.info("\nWarnings:")
                for name, message in self.warnings:
                    logger.info(f"  ⚠️ {name}: {message}")
            
            return False


def main():
    """Run system validation."""
    validator = SystemValidator()
    success = validator.run_all_tests()
    
    if success:
        print("\n🎉 System validation complete - Ready for live trading!")
        return 0
    else:
        print("\n⚠️ System validation failed - Please fix errors before trading")
        return 1


if __name__ == "__main__":
    sys.exit(main())
