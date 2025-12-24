
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_module.vertex_analyzer import get_analyzer

def test_production_analyzer():
    print("Testing VertexAnalyzer with updated settings...")
    analyzer = get_analyzer()
    if not analyzer.enabled:
        print("❌ Analyzer is DISABLED. Check logs.")
        return
    
    success = analyzer.test_connection()
    if success:
        print("✅ Vertex AI connection test successful with production code!")
    else:
        print("❌ Vertex AI connection test failed with production code.")

if __name__ == "__main__":
    test_production_analyzer()
