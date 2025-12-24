"""
Test script for Vertex AI Gemini analyzer
Tests the new AI factory with sample signal data
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from ai_module.ai_factory import get_analyzer
from config.settings import AI_PROVIDER

# Sample signal data (from your actual SHORT signal)
sample_signal = {
    "signal_type": "RESISTANCE_BOUNCE",
    "instrument": "BANKNIFTY",
    "price_level": 59001.57,
    "entry_price": 58881.50,
    "stop_loss": 59038.48,
    "take_profit": 58646.00,
    "confidence": 85,
    "risk_reward_ratio": 1.5,
    "description": "Resistance retest at 59001.57 (former support, role reversal)"
}

sample_context = {
    "trend_direction": "DOWN",
    "price_above_vwap": False,
    "price_above_ema20": False,
}

sample_technical = {}

print(f"\n{'='*70}")
print(f"🧪 Testing AI Analyzer | Provider: {AI_PROVIDER}")
print(f"{'='*70}\n")

# Get analyzer based on current config
analyzer = get_analyzer()

print(f"Analyzer: {analyzer.__class__.__name__}")
print(f"Enabled: {analyzer.enabled if hasattr(analyzer, 'enabled') else 'N/A'}\n")

# Test connection
print("Testing connection...")
if analyzer.test_connection():
    print("✅ Connection test passed\n")
else:
    print("❌ Connection test failed\n")
    sys.exit(1)

# Analyze signal
print("Analyzing sample signal...")
print(f"Signal: {sample_signal['signal_type']}")
print(f"Direction: SHORT (SL {sample_signal['stop_loss']} > Entry {sample_signal['entry_price']})")
print(f"R:R: {sample_signal['risk_reward_ratio']}\n")

result = analyzer.analyze_signal(sample_signal, sample_context, sample_technical)

if result:
    print(f"\n{'='*70}")
    print("🤖 AI Analysis Result")
    print(f"{'='*70}\n")
    print(f"Verdict: {result.get('verdict')}")
    print(f"Confidence: {result.get('confidence')}%")
    print(f"Reasoning: {result.get('reasoning')}")
    print(f"Risks: {result.get('risks')}")
    
    if 'ai_provider' in result:
        print(f"\nProvider Used: {result['ai_provider']}")
    
    # Verify direction
    verdict = result.get('verdict', '')
    if 'BUY' in verdict:
        print(f"\n❌ ERROR: Got {verdict} for SHORT signal!")
    elif 'SELL' in verdict:
        print(f"\n✅ CORRECT: Got {verdict} for SHORT signal")
    
    print(f"\n{'='*70}\n")
else:
    print("\n❌ Analysis failed - no result returned\n")

# Show usage stats
print("Usage Stats:")
stats = analyzer.get_usage_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")

print(f"\n{'='*70}\n")
