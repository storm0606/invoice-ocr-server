"""
发票 OCR 结果解析器。

从 OCR 原始文本中提取发票信息：
1. 对 OCR 结果按空间位置聚类（每张发票一个簇）
2. 每簇中提取发票号码、金额、类型
3. 用发票代码做交叉校验
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from .invoice_codes import InvoiceType, lookup_invoice_code

logger = logging.getLogger(__name__)


@dataclass
class OCRTextRegion:
    """OCR 识别出的一个文本区域"""
    text: str
    confidence: float
    box: list[list[int]]
    cx: float = 0.0
    cy: float = 0.0

    def __post_init__(self):
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        if xs and ys:
            self.cx = (min(xs) + max(xs)) / 2.0
            self.cy = (min(ys) + max(ys)) / 2.0
        else:
            self.cx = 0.0
            self.cy = 0.0


@dataclass
class RecognizedInvoice:
    """识别结果——一张发票"""
    type: InvoiceType
    amount: int
    invoice_number: str
    code: str = ""

    def to_dict(self):
        return {
            "type": self.type.value,
            "amount": self.amount,
            "invoice_number": self.invoice_number,
            "code": self.code,
        }


# ===================== 中文金额解析 =====================

CHINESE_DIGITS = {
    "壹": 1, "贰": 2, "参": 3, "叁": 3, "肆": 4,
    "伍": 5, "陆": 6, "柒": 7, "捌": 8, "玖": 9,
}

CHINESE_TENS = {
    "贰": 20, "二": 20, "叁": 30, "三": 30,
    "肆": 40, "四": 40, "伍": 50, "五": 50,
    "陆": 60, "六": 60, "柒": 70, "七": 70,
    "捌": 80, "八": 80, "玖": 90, "九": 90,
}


def parse_chinese_amount(text: str) -> Optional[int]:
    text = text.replace(" ", "")
    for suffix in ["圆", "元", "整", "正"]:
        text = text.replace(suffix, "")
    if not text:
        return None
    if "拾" in text:
        parts = text.split("拾")
        tens = CHINESE_TENS.get(parts[0], 10) if parts[0] else 10
        ones = CHINESE_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens + ones
    if text in CHINESE_DIGITS:
        return CHINESE_DIGITS[text]
    if text in CHINESE_TENS:
        return CHINESE_TENS[text]
    try:
        return int(text)
    except ValueError:
        return None


def extract_invoice_numbers(text: str) -> list[str]:
    return re.findall(r'\b(\d{8})\b', text)


def extract_invoice_codes(regions: list[OCRTextRegion]) -> list[tuple[str, int, float, float]]:
    """从文本区域中提取发票代码，返回 (code, region_index, cx, cy) 列表"""
    results = []
    for idx, region in enumerate(regions):
        m = re.search(r'[发友]票代码\s*(\d{12})', region.text)
        if m:
            code = m.group(1)
            entry = lookup_invoice_code(code)
            if entry:
                logger.info(f"  发票代码: {code} → {entry.type.value} {entry.amount}元 (位置: ({int(region.cx)}, {int(region.cy)}))")
                results.append((code, idx, region.cx, region.cy))
    return results


# ===================== Voronoi 最近邻聚类 =====================

def cluster_by_invoice_codes(regions: list[OCRTextRegion]) -> list[list[OCRTextRegion]]:
    """
    使用发票代码作为锚点，采用 Voronoi 最近邻分配进行聚类。

    策略：
    1. 找到所有发票代码的位置（锚点）
    2. 每个非锚点文本区域分配给最近的发票代码锚点
    3. 未分配到锚点的区域（远离所有锚点）按空间聚类
    """
    if not regions:
        return []

    codes = extract_invoice_codes(regions)
    if not codes:
        logger.debug("  未找到发票代码，使用纯空间聚类")
        return cluster_spatial(regions)

    code_indices = {c[1] for c in codes}
    code_regions_list = [(c[0], c[2], c[3]) for c in codes]

    clusters = {i: [] for i in range(len(codes))}

    # 先把代码本身加入对应的簇
    for ci, (code, idx, cx, cy) in enumerate(codes):
        clusters[ci].append(regions[idx])

    # Voronoi 分配：每个非锚点区域分配给最近的代码锚点
    for idx, region in enumerate(regions):
        if idx in code_indices:
            continue
        min_dist = float('inf')
        nearest = -1
        for ci, (code, cx, cy) in enumerate(code_regions_list):
            dx = region.cx - cx
            dy = region.cy - cy
            dist = dx * dx + dy * dy
            if dist < min_dist:
                min_dist = dist
                nearest = ci
        if nearest >= 0:
            clusters[nearest].append(region)

    # 合并结果
    result = []
    for ci in range(len(codes)):
        result.append(clusters[ci])
        logger.debug(f"  code_cluster[{ci}]: {len(clusters[ci])} regions (code: {codes[ci][0]})")

    logger.info(f"  聚类: {len(codes)} 个代码锚点, 共 {len(result)} 个簇, "
                f"{sum(len(c) for c in result)}/{len(regions)} 文本区域")

    return result


def cluster_spatial(regions: list[OCRTextRegion]) -> list[list[OCRTextRegion]]:
    """
    纯空间聚类：先按 Y 行分割，再按 X 列分割。
    适用于发票被摆成表格/网格布局的场景。
    """
    if not regions:
        return []
    if len(regions) <= 3:
        return [regions]

    entries = list(regions)
    all_ys = [e.cy for e in entries]
    all_xs = [e.cx for e in entries]
    image_height = max(all_ys) - min(all_ys) + 1 if all_ys else 1
    image_width = max(all_xs) - min(all_xs) + 1 if all_xs else 1

    sorted_by_cy = sorted(entries, key=lambda e: e.cy)
    vertical_gaps = [
        sorted_by_cy[i].cy - sorted_by_cy[i - 1].cy
        for i in range(1, len(sorted_by_cy))
    ]
    sorted_vg = sorted(vertical_gaps)
    median_vgap = sorted_vg[len(sorted_vg) // 2] if sorted_vg else 0
    row_gap_threshold = max(median_vgap * 3, image_height * 0.05, 60.0)

    logger.debug(f"  空间聚类: {len(regions)} regions, "
                 f"medianVGap={median_vgap:.0f} rowThreshold={row_gap_threshold:.0f}")

    rows = []
    current_row = [sorted_by_cy[0]]
    for i in range(1, len(sorted_by_cy)):
        gap = sorted_by_cy[i].cy - sorted_by_cy[i - 1].cy
        if gap > row_gap_threshold:
            rows.append(current_row)
            current_row = []
        current_row.append(sorted_by_cy[i])
    if current_row:
        rows.append(current_row)

    clusters = []
    for row in rows:
        sorted_by_cx = sorted(row, key=lambda e: e.cx)
        horiz_gaps = [
            sorted_by_cx[i].cx - sorted_by_cx[i - 1].cx
            for i in range(1, len(sorted_by_cx))
        ]
        sorted_hg = sorted(horiz_gaps) if horiz_gaps else [0]
        median_hgap = sorted_hg[len(sorted_hg) // 2] if sorted_hg else 0
        col_gap_threshold = max(median_hgap * 3, image_width * 0.06, 50.0)

        current_col = [sorted_by_cx[0]]
        for i in range(1, len(sorted_by_cx)):
            gap = sorted_by_cx[i].cx - sorted_by_cx[i - 1].cx
            if gap > col_gap_threshold:
                clusters.append(current_col)
                current_col = []
            current_col.append(sorted_by_cx[i])
        clusters.append(current_col)

    return clusters


# ===================== 一卡通检测 =====================

def is_yikatong_by_features(cluster: list[OCRTextRegion]) -> tuple[bool, str | None]:
    """基于特征检测是否为一卡通发票"""
    full_text = " ".join(r.text for r in cluster)
    if "加盖收款方" in full_text or "发票专用章有效" in full_text:
        return True, "has '加盖收款方发票专用章有效'"
    if "天津通用定额发票" in full_text:
        return True, "has '天津通用定额发票'"
    if "¥" in full_text or "￥" in full_text:
        return False, "has ¥ symbol"
    for region in cluster:
        for other in cluster:
            if other is region:
                continue
            if abs(other.cy - region.cy) < 150:
                if "¥" in other.text or "￥" in other.text:
                    return False, f"nearby ¥ found: {other.text}"
    return False, "unknown"


# ===================== 主解析逻辑 =====================

def parse_invoices(ocr_results: list[dict]) -> list[dict]:
    regions = []
    for r in ocr_results:
        regions.append(OCRTextRegion(
            text=r["text"],
            confidence=r["confidence"],
            box=r["box"],
        ))

    if not regions:
        logger.warning("OCR 结果为空")
        return []

    clusters = cluster_by_invoice_codes(regions)

    all_invoices = []
    for cluster in clusters:
        inv = parse_single_cluster(cluster)
        if inv:
            all_invoices.append(inv)

    all_invoices = deduplicate_invoices(all_invoices)

    logger.info(f"解析完成: {len(all_invoices)} 张发票")
    for inv in all_invoices:
        logger.info(f"  {inv.type.value} {inv.amount}元 号码={inv.invoice_number} "
                    f"code={inv.code} ✓")

    return [inv.to_dict() for inv in all_invoices]


def parse_single_cluster(cluster: list[OCRTextRegion]) -> Optional[RecognizedInvoice]:
    """解析单个簇，提取发票信息"""
    full_text = " ".join(r.text for r in cluster)

    # Step 0: 从发票代码确定类型和金额（最可靠的方式）
    inv_type = None
    amount = None
    invoice_code = ""

    for region in cluster:
        m = re.search(r'[发友]票代码\s*(\d{12})', region.text)
        if m:
            code = m.group(1)
            entry = lookup_invoice_code(code)
            if entry:
                invoice_code = code
                inv_type = entry.type
                amount = entry.amount
                logger.debug(f"  代码确定: {entry.type.value} {entry.amount}元 (code={code})")
                break

    # Step 1: 提取发票号码
    numbers = extract_invoice_numbers(full_text)
    if not numbers:
        logger.debug(f"  未找到号码: {full_text[:80]}")
        return None

    inv_number = numbers[0]

    # 优先用发票代码所在区域的号码（如果有的话）
    if invoice_code:
        for region in cluster:
            if invoice_code in region.text:
                nums_in_region = extract_invoice_numbers(region.text)
                for n in nums_in_region:
                    if len(n) == 8 and n != invoice_code:
                        inv_number = n
                        break

    # Step 2: 确定发票类型（代码没提供时用特征）
    if inv_type is None:
        is_ykt, _ = is_yikatong_by_features(cluster)
        inv_type = InvoiceType.YIKATONG if is_ykt else InvoiceType.REGULAR

    # Step 3: 确定金额（代码没提供时从文本提取）
    if amount is None:
        amount = extract_amount_from_cluster(cluster, inv_type)

    if amount is None:
        logger.debug(f"  未找到金额: {full_text[:80]}")
        return None

    return RecognizedInvoice(
        type=inv_type,
        amount=amount,
        invoice_number=inv_number,
        code=invoice_code,
    )


def extract_amount_from_cluster(cluster: list[OCRTextRegion], inv_type: InvoiceType) -> Optional[int]:
    """从簇文本中提取金额"""
    full_text = " ".join(r.text for r in cluster)

    # 普通发票通常有 ¥/￥ 符号
    if inv_type == InvoiceType.REGULAR:
        m = re.search(r'[¥￥]\s*(\d+(?:\.\d+)?)', full_text)
        if m:
            try:
                amt = int(float(m.group(1)))
                if 1 <= amt <= 200:
                    return amt
            except ValueError:
                pass

    # 中文大写金额
    for suffix in ["圆", "元", "拾"]:
        idx = full_text.find(suffix)
        if idx >= 0:
            start = max(0, idx - 10)
            end = min(len(full_text), idx + 5)
            amt = parse_chinese_amount(full_text[start:end])
            if amt and 1 <= amt <= 200:
                return amt

    # 逐区域尝试
    for r in cluster:
        amt = parse_chinese_amount(r.text)
        if amt and 1 <= amt <= 200:
            return amt

    return None


def deduplicate_invoices(invoices: list[RecognizedInvoice]) -> list[RecognizedInvoice]:
    """
    去重：同一类型+金额只保留一个。
    因为每次拍照每种金额只有一张发票。
    保留优先使用发票代码确定的发票（更准确）。
    """
    sorted_invoices = sorted(invoices, key=lambda x: (0 if x.code else 1, x.type.value, x.amount))
    seen = set()
    result = []
    for inv in sorted_invoices:
        key = (inv.type, inv.amount)
        if key not in seen:
            seen.add(key)
            result.append(inv)
        else:
            logger.info(f"  去重: {inv.type.value} {inv.amount}元 (号码 {inv.invoice_number})")
    return result
