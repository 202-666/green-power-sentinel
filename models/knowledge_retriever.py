"""
绿电哨兵 — Agent 4 知识库检索引擎
实现技术框架 §4.4「检索逻辑」：
  1. 关键词匹配：基于 fault_type / fault_subtype / symptom_pattern 精确与关键词匹配
  2. 语义相似度：使用 TF-IDF + 余弦相似度（离线，规避对豆包 API 的强依赖）
  3. 排序：按综合匹配度从高到低返回 Top-K 案例

三维加权评分：
  score = w_type * type_match + w_kw * keyword_match + w_sem * semantic_sim
  默认权重：0.40 / 0.30 / 0.30
"""

import logging
import os
import re
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# 检索权重（可被配置覆盖）
DEFAULT_WEIGHTS = {"type": 0.40, "keyword": 0.30, "semantic": 0.30}

# 中文分词近似：以标点与空白切分，并提取参数名 token
# 症状模式形如 "bearing_temperature↑ AND bearing_vibration↑"
_TOKEN_SEP = re.compile(r"[,\s、，;；]+")
_DIR_ARROW = re.compile(r"[↑↓波动]+")
# 参数名 + 方向 模式，用于提取带方向的参数 token
_PARAM_WITH_DIR = re.compile(
    r"([a-zA-Z_][a-zA-Z0-9_]*)\s*([↑↓]|波动)"
)
_LOGIC_WORDS = {"AND", "OR", "and", "or", "且", "或"}


def _normalize_tokens(text: str) -> list:
    """
    将症状模式或描述文本切分为标准化 token 列表。
    去除方向箭头与逻辑连接词，仅保留参数名与关键词。
    """
    if not text:
        return []
    # 移除方向/波动箭头
    cleaned = _DIR_ARROW.sub(" ", str(text))
    # 替换逻辑连接词为分隔符
    for w in _LOGIC_WORDS:
        cleaned = cleaned.replace(w, " ")
    # 按分隔符切分
    parts = _TOKEN_SEP.split(cleaned)
    return [p.strip() for p in parts if p.strip()]


def _extract_params_with_direction(text: str) -> dict:
    """
    从症状模式文本中提取「参数名 → 方向」映射。
    方向取值："up" (↑) / "down" (↓) / "volatile" (波动) / "unknown"

    例："bearing_temperature↑ AND bearing_vibration↑"
    → {"bearing_temperature": "up", "bearing_vibration": "up"}
    """
    if not text:
        return {}
    result = {}
    dir_map = {"↑": "up", "↓": "down", "波动": "volatile"}
    for m in _PARAM_WITH_DIR.finditer(str(text)):
        param = m.group(1)
        arrow = m.group(2)
        result[param] = dir_map.get(arrow, "unknown")
    return result


class KnowledgeRetriever:
    """故障知识库检索引擎"""

    def __init__(
        self,
        kb_path: str = "config/knowledge_base.yaml",
        weights: Optional[dict] = None,
    ):
        """
        初始化检索引擎，加载知识库并构建 TF-IDF 索引。

        Args:
            kb_path: 知识库 YAML 路径
            weights: 检索权重 {type, keyword, semantic}，None 则使用默认权重
        """
        if not os.path.exists(kb_path):
            raise FileNotFoundError(f"Knowledge base not found: {kb_path}")

        with open(kb_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.cases = data.get("cases", [])
        if not self.cases:
            raise ValueError(f"Knowledge base is empty: {kb_path}")

        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self._build_index()
        logger.info(
            f"KnowledgeRetriever initialized: {len(self.cases)} cases, "
            f"weights={self.weights}"
        )

    def _build_index(self):
        """
        构建 TF-IDF 索引（离线语义相似度）。
        文档 = symptom_pattern + fault_subtype + description 的拼接文本。
        使用 jieba 可选分词，未安装时退化为字符级 n-gram。
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for semantic similarity. "
                "pip install scikit-learn"
            ) from e

        self._cosine_similarity = cosine_similarity

        # 构造每条案例的文档文本
        docs = []
        for c in self.cases:
            doc = " ".join([
                str(c.get("fault_type", "")),
                str(c.get("fault_subtype", "")),
                str(c.get("symptom_pattern", "")),
                str(c.get("description", "")),
            ])
            docs.append(self._tokenize_for_tfidf(doc))
        self._docs = docs

        # 字符级 n-gram（1-2 gram），适配中文无分词场景
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(docs)
        logger.debug(f"TF-IDF index built: shape={self._tfidf_matrix.shape}")

    @staticmethod
    def _tokenize_for_tfidf(text: str) -> str:
        """
        文本预处理为 TF-IDF 可用形式。
        优先使用 jieba 分词（如安装），否则按标点/空白切分并保留参数名。
        """
        try:
            import jieba
            return " ".join(jieba.cut(str(text)))
        except ImportError:
            # 退化方案：按标点与空白切分，保留 CJK 字符片段
            tokens = _normalize_tokens(text)
            # 补充：将长串 CJK 文本按 2 字滑动切分以增强语义捕获
            extra = []
            for t in tokens:
                if len(t) >= 2:
                    # 对纯中文 token 追加 2-gram
                    cjk_only = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9_]+", "", t)
                    if len(cjk_only) >= 4:
                        extra += [cjk_only[i:i + 2] for i in range(len(cjk_only) - 1)]
            return " ".join(tokens + extra)

    # ==================================================================
    # 三维评分
    # ==================================================================

    def _type_match_score(self, case: dict, fault_type: str) -> float:
        """故障类型精确匹配：完全相同=1.0，缺省查询=0.5，不匹配=0.0"""
        if not fault_type:
            return 0.5  # 无明确故障类型时不惩罚
        case_type = str(case.get("fault_type", ""))
        if case_type == fault_type:
            return 1.0
        # 容错：包含关系（如查询"轴承过热"，案例为"综合故障"但子类相关）
        if fault_type in case_type or case_type in fault_type:
            return 0.6
        # 未匹配故障类型但属于综合故障案例，给予弱召回
        if case_type == "综合故障":
            return 0.3
        return 0.0

    def _keyword_match_score(self, case: dict, symptom_pattern: str,
                             param_values: Optional[dict] = None,
                             query_dir_map: Optional[dict] = None) -> float:
        """
        症状模式关键词匹配（含方向感知）。
        基础分 = 参数名覆盖率 × 0.7 + Jaccard × 0.3
        方向加成 = 方向一致的参数比例 × 0.15（额外加分，不超过基础分上限）
        方向惩罚 = 方向相反的参数比例 × 0.15（从基础分中扣减）
        """
        query_tokens = set(_normalize_tokens(symptom_pattern or ""))
        if param_values:
            query_tokens |= set(str(k) for k in param_values.keys())

        if not query_tokens:
            return 0.5

        case_tokens = set(_normalize_tokens(case.get("symptom_pattern", "")))
        if not case_tokens:
            return 0.0

        hit = query_tokens & case_tokens
        coverage = len(hit) / len(query_tokens)
        jaccard = len(hit) / len(query_tokens | case_tokens)
        base_score = 0.7 * coverage + 0.3 * jaccard

        # 方向感知：如果查询和案例都有方向信息，则计算方向一致性
        if query_dir_map:
            case_dir_map = _extract_params_with_direction(case.get("symptom_pattern", ""))
            if case_dir_map:
                common_params = set(query_dir_map.keys()) & set(case_dir_map.keys())
                if common_params:
                    same_dir = sum(
                        1 for p in common_params
                        if query_dir_map[p] == case_dir_map[p]
                    )
                    opp_dir = sum(
                        1 for p in common_params
                        if query_dir_map[p] in ("up", "down")
                        and case_dir_map[p] in ("up", "down")
                        and query_dir_map[p] != case_dir_map[p]
                    )
                    same_ratio = same_dir / len(common_params)
                    opp_ratio = opp_dir / len(common_params)
                    # 方向一致最高 +0.15，相反最高 -0.15
                    base_score = base_score + 0.15 * same_ratio - 0.15 * opp_ratio
                    base_score = max(0.0, min(1.0, base_score))

        return base_score

    def _compute_semantic_sims(self, query_text: str):
        """
        一次性计算查询文本与所有案例的 TF-IDF 余弦相似度。
        （性能优化：避免循环中重复 transform）
        """
        if not query_text.strip():
            return [0.0] * len(self.cases)
        query_vec = self._vectorizer.transform([self._tokenize_for_tfidf(query_text)])
        sims = self._cosine_similarity(query_vec, self._tfidf_matrix).ravel()
        return [max(0.0, float(s)) for s in sims]

    # ==================================================================
    # 公共检索接口
    # ==================================================================

    def retrieve(
        self,
        fault_type: str,
        fault_subtype: Optional[str] = None,
        symptom_pattern: Optional[str] = None,
        param_values: Optional[dict] = None,
        top_k: int = 3,
    ) -> list:
        """
        检索 Top-K 最相似案例。

        Args:
            fault_type: 故障类型（如 "轴承过热"）
            fault_subtype: 故障子类（可选）
            symptom_pattern: 症状模式（参数组合，如 "bearing_temperature↑ AND bearing_vibration↑"）
            param_values: 异常参数当前值 {param: value}，用于补充症状 token
            top_k: 返回前 K 条

        Returns:
            排序后的案例列表，每条附加 match_score / match_details 字段
        """
        # 构造语义查询文本
        query_parts = [fault_type or "", fault_subtype or "", symptom_pattern or ""]
        if param_values:
            query_parts.append(" ".join(str(k) for k in param_values.keys()))
        query_text = " ".join(p for p in query_parts if p)

        # 预计算：方向映射 + 语义相似度（性能优化，循环外一次计算）
        query_dir_map = _extract_params_with_direction(symptom_pattern or "")
        sem_sims = self._compute_semantic_sims(query_text)

        scored = []
        w = self.weights
        for idx, case in enumerate(self.cases):
            type_s = self._type_match_score(case, fault_type)
            kw_s = self._keyword_match_score(
                case, symptom_pattern, param_values, query_dir_map
            )
            sem_s = sem_sims[idx] if idx < len(sem_sims) else 0.0

            # 子类精确匹配：从 keyword 权重池中分配 0.05（而非额外叠加）
            subtype_bonus = 0.0
            if fault_subtype and case.get("fault_subtype") == fault_subtype:
                subtype_bonus = 0.05  # 归入 keyword 维度的额外奖励

            total = (
                w["type"] * type_s
                + w["keyword"] * (kw_s + subtype_bonus)
                + w["semantic"] * sem_s
            )
            total = max(0.0, min(1.0, total))

            scored.append({
                **case,
                "match_score": round(total, 4),
                "match_details": {
                    "type_match": round(type_s, 4),
                    "keyword_match": round(kw_s, 4),
                    "semantic_sim": round(sem_s, 4),
                    "subtype_bonus": round(subtype_bonus, 4),
                },
            })

        scored.sort(key=lambda c: c["match_score"], reverse=True)
        top = scored[:top_k]
        logger.info(
            f"Retrieve fault_type='{fault_type}' symptom='{symptom_pattern}' "
            f"→ top{top_k}: {[(c['case_id'], c['match_score']) for c in top]}"
        )
        return top

    def get_case_by_id(self, case_id: str) -> Optional[dict]:
        """按 case_id 精确查找案例"""
        for c in self.cases:
            if c.get("case_id") == case_id:
                return c
        return None

    def stats(self) -> dict:
        """知识库统计信息"""
        from collections import Counter
        type_dist = Counter(c.get("fault_type", "") for c in self.cases)
        return {
            "total_cases": len(self.cases),
            "by_fault_type": dict(type_dist),
            "weights": self.weights,
        }
