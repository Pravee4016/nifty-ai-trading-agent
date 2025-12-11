# Fyers OAuth Token Manager
# Handles automatic token refresh using OAuth refresh tokens

import logging
import os
import json
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import access_token
import requests

logger = logging.getLogger(__name__)


class FyersOAuthManager:
    """
    Manages Fyers OAuth tokens with automatic refresh capability.
    Stores refresh token in Cloud Secret Manager for persistence.
    """
    
    def __init__(self, app_id: str, secret_key: str, redirect_uri: str = "https://trade.fyers.in/api-login/redirect-uri/index.html"):
        self.app_id = app_id
        self.secret_key = secret_key
        self.redirect_uri = redirect_uri
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        
        # Try to load existing tokens
        self._load_tokens()
    
    def _load_tokens(self):
        """Load tokens from environment or Cloud Secret Manager."""
        # Try environment first (for local development)
        self.access_token = os.getenv("FYERS_ACCESS_TOKEN")
        self.refresh_token = os.getenv("FYERS_REFRESH_TOKEN")
        
        # Try Cloud Secret Manager if in production
        if not self.refresh_token:
            self.refresh_token = self._load_from_secret_manager("fyers-refresh-token")
        
        if self.refresh_token:
            logger.info("✅ Loaded refresh token from storage")
    
    def _load_from_secret_manager(self, secret_name: str) -> Optional[str]:
        """Load secret from Google Cloud Secret Manager."""
        try:
            from google.cloud import secretmanager
            
            project_id = os.getenv("GCP_PROJECT_ID", "nifty-trading-agent")
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            
            response = client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            
            logger.info(f"✅ Loaded {secret_name} from Secret Manager")
            return secret_value
        except Exception as e:
            logger.debug(f"Could not load {secret_name} from Secret Manager: {e}")
            return None
    
    def _save_to_secret_manager(self, secret_name: str, secret_value: str) -> bool:
        """Save secret to Google Cloud Secret Manager."""
        try:
            from google.cloud import secretmanager
            
            project_id = os.getenv("GCP_PROJECT_ID", "nifty-trading-agent")
            client = secretmanager.SecretManagerServiceClient()
            parent = f"projects/{project_id}"
            secret_id = secret_name
            
            # Check if secret exists
            try:
                secret_path = f"{parent}/secrets/{secret_id}"
                client.get_secret(request={"name": secret_path})
                exists = True
            except:
                exists = False
            
            # Create secret if it doesn't exist
            if not exists:
                client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": secret_id,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )
            
            # Add new version
            parent_path = f"{parent}/secrets/{secret_id}"
            payload = secret_value.encode("UTF-8")
            client.add_secret_version(
                request={"parent": parent_path, "payload": {"data": payload}}
            )
            
            logger.info(f"✅ Saved {secret_name} to Secret Manager")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save {secret_name} to Secret Manager: {e}")
            return False
    
    def generate_auth_url(self) -> str:
        """
        Generate Fyers authorization URL for initial OAuth flow.
        User needs to visit this URL and authorize the app.
        """
        session_model = fyersModel.SessionModel(
            client_id=self.app_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            response_type="code",
            grant_type="authorization_code"
        )
        
        auth_url = session_model.generate_authcode()
        logger.info(f"🔐 Authorization URL generated")
        return auth_url
    
    def exchange_auth_code(self, auth_code: str) -> Tuple[bool, str]:
        """
        Exchange authorization code for access token and refresh token.
        
        Args:
            auth_code: The authorization code from Fyers OAuth callback
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            session_model = fyersModel.SessionModel(
                client_id=self.app_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type="code",
                grant_type="authorization_code"
            )
            
            session_model.set_token(auth_code)
            response = session_model.generate_token()
            
            if response.get("code") == 200:
                self.access_token = response.get("access_token")
                self.refresh_token = response.get("refresh_token")
                
                # Calculate expiry (Fyers tokens typically last 24 hours)
                self.token_expiry = datetime.now() + timedelta(hours=23, minutes=50)
                
                # Save refresh token to Secret Manager for persistence
                self._save_to_secret_manager("fyers-refresh-token", self.refresh_token)
                
                logger.info("✅ Successfully exchanged auth code for tokens")
                return True, "Tokens obtained successfully"
            else:
                error_msg = response.get("message", "Unknown error")
                logger.error(f"❌ Token exchange failed: {error_msg}")
                return False, error_msg
                
        except Exception as e:
            logger.error(f"❌ Exception during token exchange: {e}")
            return False, str(e)
    
    def refresh_access_token(self) -> Tuple[bool, str]:
        """
        Refresh the access token using the refresh token.
        This is called automatically when the access token expires.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.refresh_token:
            logger.error("❌ No refresh token available")
            return False, "No refresh token available"
        
        try:
            session_model = fyersModel.SessionModel(
                client_id=self.app_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type="code",
                grant_type="refresh_token"
            )
            
            session_model.set_token(self.refresh_token)
            response = session_model.generate_token()
            
            if response.get("code") == 200:
                self.access_token = response.get("access_token")
                
                # Update refresh token if a new one is provided
                if response.get("refresh_token"):
                    self.refresh_token = response.get("refresh_token")
                    self._save_to_secret_manager("fyers-refresh-token", self.refresh_token)
                
                # Update expiry
                self.token_expiry = datetime.now() + timedelta(hours=23, minutes=50)
                
                logger.info("✅ Access token refreshed successfully")
                return True, "Token refreshed successfully"
            else:
                error_msg = response.get("message", "Unknown error")
                logger.error(f"❌ Token refresh failed: {error_msg}")
                return False, error_msg
                
        except Exception as e:
            logger.error(f"❌ Exception during token refresh: {e}")
            return False, str(e)
    
    def get_valid_access_token(self) -> Optional[str]:
        """
        Get a valid access token, automatically refreshing if needed.
        This is the main method to call when you need an access token.
        
        Returns:
            Valid access token or None if refresh failed
        """
        # Check if we have an access token
        if not self.access_token:
            logger.warning("⚠️ No access token available")
            
            # Try to refresh if we have a refresh token
            if self.refresh_token:
                logger.info("🔄 Attempting to refresh access token...")
                success, message = self.refresh_access_token()
                if success:
                    return self.access_token
                else:
                    logger.error(f"❌ Token refresh failed: {message}")
                    return None
            else:
                logger.error("❌ No refresh token available - need to re-authorize")
                return None
        
        # Check if token is expired or about to expire (within 10 minutes)
        if self.token_expiry and datetime.now() >= self.token_expiry - timedelta(minutes=10):
            logger.info("🔄 Access token expired or expiring soon, refreshing...")
            success, message = self.refresh_access_token()
            if not success:
                logger.warning(f"⚠️ Token refresh failed: {message}, using existing token")
        
        return self.access_token
    
    def is_authorized(self) -> bool:
        """Check if we have valid authorization (refresh token available)."""
        return self.refresh_token is not None


# Singleton instance
_oauth_manager: Optional[FyersOAuthManager] = None


def get_oauth_manager() -> FyersOAuthManager:
    """Get or create the OAuth manager singleton."""
    global _oauth_manager
    
    if _oauth_manager is None:
        app_id = os.getenv("FYERS_CLIENT_ID", "")
        secret_key = os.getenv("FYERS_SECRET_ID", "")
        
        if not app_id or not secret_key:
            raise ValueError("FYERS_CLIENT_ID and FYERS_SECRET_ID must be set")
        
        _oauth_manager = FyersOAuthManager(app_id, secret_key)
    
    return _oauth_manager
