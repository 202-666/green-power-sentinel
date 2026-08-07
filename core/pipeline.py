"""
绿电哨兵 — 四Agent流水线调度器
负责协调数据采集、异常检测、预警推送、运维建议四个Agent
"""

import yaml
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentinelPipeline:
    """四Agent流水线调度器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._cached_df = None
        self._data_source_path = None
        self._detection_scores = None
        self._detection_df = None
        self._last_anomalies = []
        self._yellow_tracker = self._load_yellow_tracker()
        self._init_modules()
        logger.info(f"SentinelPipeline initialized, mode={self.config.get('app', {}).get('mode', 'unknown')}")

    def _load_config(self) -> dict:
        """加载配置文件，支持环境变量解析"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config = self._resolve_env_vars(config)
        logger.debug(f"Config loaded from {self.config_path}")
        return config

    def _resolve_env_vars(self, obj):
        """递归解析配置中的环境变量，格式: ${ENV_VAR:-default}"""
        import re
        pattern = re.compile(r'\$\{([^}]+)\}')

        def _resolve(value):
            if isinstance(value, str):
                matches = pattern.findall(value)
                for match in matches:
                    parts = match.split(':-', 1)
                    env_name = parts[0].strip()
                    default = parts[1].strip() if len(parts) > 1 else ''
                    env_value = os.environ.get(env_name, default)
                    value = value.replace(f'${{{match}}}', env_value)
                return value
            elif isinstance(value, dict):
                return {k: _resolve(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_resolve(item) for item in value]
            else:
                return value

        return _resolve(obj)

    def _yellow_tracker_path(self) -> str:
        """黄色升级计数器的持久化路径（L7：进程重启不清零）。

        路径固定于 ~/.aily/workspace/yellow_tracker.json
        （跨 run 持久化必备，确保 auto 定时任务间状态不丢失）。
        """
        import os.path
        return os.path.expanduser("~/.aily/workspace/yellow_tracker.json")

    def _load_yellow_tracker(self) -> dict:
        """启动时从本地 JSON 加载黄色升级计数器。"""
        import json

        path = self._yellow_tracker_path()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {
                        str(k): int(v)
                        for k, v in data.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    }
        except Exception as e:
            logger.warning(f"黄色升级计数器加载失败，使用空状态: {e}")
        return {}

    def _save_yellow_tracker(self) -> None:
        """将黄色升级计数器持久化到本地 JSON（失败仅告警，不阻断流水线）。"""
        import json

        try:
            path = self._yellow_tracker_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._yellow_tracker, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"黄色升级计数器持久化失败: {e}")
