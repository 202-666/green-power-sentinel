import requests
import time

# L4：飞书鉴权请求超时（秒），并允许一次重试
REQUEST_TIMEOUT = 10
MAX_ATTEMPTS = 2


class FeishuAuth:
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.token = None
        self.token_expire_time = 0

    def get_tenant_access_token(self) -> str:
        if self.token and time.time() < self.token_expire_time:
            return self.token
        
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        
        last_err = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") == 0:
                    self.token = data["tenant_access_token"]
                    self.token_expire_time = time.time() + data.get("expire", 7200) - 60
                    return self.token
                raise Exception(f"Failed to get token: {data.get('msg')}")
            except Exception as e:
                last_err = e
                if attempt < MAX_ATTEMPTS:
                    time.sleep(1)
        raise Exception(f"Token request failed: {str(last_err)}")
