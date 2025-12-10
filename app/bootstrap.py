"""
Bootstrap Module
Handles the creation and initialization of the Trading Agent.
Implements the AppFactory pattern.
"""
import logging
from app.agent import NiftyTradingAgent
from config.logging_config import setup_logging

logger = setup_logging(__name__)

def create_agent() -> NiftyTradingAgent:
    """
    Factory function to create and configure the trading agent.
    """
    try:
        logger.info("🔧 Bootstrapping NiftyTradingAgent...")
        agent = NiftyTradingAgent()
        return agent
    except Exception as e:
        logger.critical(f"❌ Failed to create agent: {e}", exc_info=True)
        raise
