"""
发票 OCR 结果解析器。

从 OCR 原始文本中提取发票信息：
1. 对 OCR 结果按空间位置聚类（每张发票一个簇）
2. 每簇中提取发票号码、金额、类型
3. 用发票代码做交叉校验
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .invoice_codes import InvoiceType, lookup_invoice_code

logger = logging.getLogger(__name__)

# ===================== 数据类 =====================

@dataclass
class OCRTextRegion:
    """OCR 识别出的一个文本区域"""
    text: str
    confidence: float
    box: list[list[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    cx: float = 0.0       # 中心 X
    cy: float = 0.0       # 中心 Y

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
    corrected_by_code: bool = False

    def to_dict(self):
        return {
            "type": self.type.value,
            "amount": self.amount,
            "invoice_number": self.invoice_number,
            "corrected_by_code": self.corrected_by_code,
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
    """
    解析中文大写金额，如 "贰圆" → 2, "壹拾圆" → 10, "伍拾圆" → 50。
    """
    text = text.replace(" ", "")

    # 去掉 "圆"、"元"、"整" 等后缀
    for suffix in ["圆", "元", "整", "正"]:
        text = text.replace(suffix, "")

    if not text:
        return None

    # 匹配 "XX拾X" 或 "拾X" 格式的金额
    if "拾" in text:
        parts = text.split("拾")
        tens = CHINESE_TENS.get(parts[0], 10) if parts[0] else 10
        ones = CHINESE_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens + ones

    # 匹配单个数字（1-9）
    if text in CHINESE_DIGITS:
        return CHINESE_DIGITS[text]

    # 尝试匹配 "贰"、"伍" 等作为十位数
    if text in CHINESE_TENS:
        return CHINESE_TENS[text]

    # 尝试匹配纯数字
    try:
        return int(text)
    except ValueError:
        return None


# ===================== 号码提取 =====================

def extract_invoice_numbers(text: str) -> list[str]:
    """
    从文本中提取 8 位发票号码。
    发票号码格式：图片上一般显示为 8 位数字。
    """
    # 寻找 8 位数字（发票号码通常是 8 位）
    matches = re.findall(r'\b(\d{8})\b', text)
    return matches


def extract_invoice_codes(ocr_results: list[OCRTextRegion]) -> list[tuple[str, float, float]]:
    """
    从 OCR 结果中提取所有发票代码（带位置信息）。
    发票代码格式："发票代码" + 12 位数字。

    Returns:
        list of (code, cx, cy)
    """
    results = []
    for region in ocr_results:
        # 匹配 "发票代码" + 数字，OCR 可能把 "发" 识别成 "友"
        m = re.search(r'[发友]票代码\s*(\d{12})', region.text)
        if m:
            code = m.group(1)
            entry = lookup_invoice_code(code)
            if entry:
                logger.info(f"  发票代码: {code} → {entry.type.value} {entry.amount}元")
                results.append((code, region.cx, region.cy))
    return results


# ===================== 聚类 =====================

def cluster_ocr_results(results: list[OCRTextRegion]) -> list[list[OCRTextRegion]]:
    """
    将 OCR 结果按 2D 空间聚类（先按 Y 分列，再按 X 分行）。
    """
    if len(results) <= 3:
        return [results]

    entries = list(results)
    all_ys = [e.cy for e in entries]
    all_xs = [e.cx for e in entries]
    image_height = max(all_ys) - min(all_ys) + 1
    image_width = max(all_xs) - min(all_xs) + 1

    # ---- 第一步：按 Y 质心拆分为行 ----
    sorted_by_cy = sorted(entries, key=lambda e: e.cy)
    vertical_gaps = [
        sorted_by_cy[i].cy - sorted_by_cy[i - 1].cy
        for i in range(1, len(sorted_by_cy))
    ]
    sorted_vg = sorted(vertical_gaps)
    median_vgap = sorted_vg[len(sorted_vg) // 2] if sorted_vg else 0
    row_gap_threshold = max(median_vgap * 3, image_height * 0.05, 60.0)

    logger.debug(f"  聚类: {len(results)} regions, "
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

    logger.debug(f"  行数: {len(rows)}")

    # ---- 第二步：每行内按 X 质心拆分为列 ----
    clusters = []
    for row in rows:
        sorted_by_cx = sorted(row, key=lambda e: e.cx)
        horiz_gaps = [
            sorted_by_cx[i].cx - sorted_by_cx[i - 1].cx
            for i in range(1, len(sorted_by_cx))
        ]
        sorted_hg = sorted(horiz_gaps)
        median_hgap = sorted_hg[len(sorted_hg) // 2] if sorted_hg else 0
        col_gap_threshold = max(median_hgap * 3, image_width * 0.06, 50.0)

        logger.debug(f"    行内 {len(row)} regions, "
                     f"medianHGap={median_hgap:.0f} colThreshold={col_gap_threshold:.0f}")

        current_col = [sorted_by_cx[0]]
        for i in range(1, len(sorted_by_cx)):
            gap = sorted_by_cx[i].cx - sorted_by_cx[i - 1].cx
            if gap > col_gap_threshold:
                clusters.append(current_col)
                current_col = []
            current_col.append(sorted_by_cx[i])
        clusters.append(current_col)

    logger.debug(f"  总簇数: {len(clusters)}")
    for i, c in enumerate(clusters):
        texts = " ".join(r.text for r in c)
        logger.debug(f"    cluster[{i}]: {len(c)} regions: {texts[:80]}")
        for r in c:
            logger.debug(f"      [{r.text}] box={r.box}")

    return clusters


# ===================== Y 一卡通检测 =====================

def is_yikatong_by_features(cluster: list[OCRTextRegion]) -> tuple[bool, str | None]:
    """
    通过 OCR 特征判断是否为一卡通发票。

    一卡通特征：
    1. 发票头部是"天津通用定额发票"
    2. 没有小写金额如 "¥ 2.00"（普通发票才有）
    3. 大写金额底下有一行字"加盖收款方发票专用章有效"
    4. 右下角有二维码

    Returns:
        (is_yikatong, reason)
    """
    full_text = " ".join(r.text for r in cluster)

    # 特征 3：有"加盖收款方发票专用章有效" → 一卡通
    if "加盖收款方" in full_text or "发票专用章有效" in full_text:
        return True, "has '加盖收款方发票专用章有效'"

    # 特征 1：头部是"天津通用定额发票" → 一卡通
    if "天津通用定额发票" in full_text:
        return True, "has '天津通用定额发票'"

    # 特征 2：有 ¥ 或 ￥ → 普通发票（不是一卡通）
    if "¥" in full_text or "￥" in full_text:
        return False, "has ¥ symbol"

    # 检查附近区域是否有 ¥（一卡通区域附近不会有 ¥）
    for region in cluster:
        center_y = region.cy
        for other in cluster:
            if other is region:
                continue
            if abs(other.cy - center_y) < 150:
                if "¥" in other.text or "￥" in other.text:
                    return False, f"nearby ¥ found: {other.text}"

    # 没有明确特征，按发票代码判断（如果有的话）
    return False, "unknown"


# ===================== 主解析逻辑 =====================

def parse_invoices(ocr_results: list[dict]) -> list[dict]:
    """
    解析 OCR 原始结果，提取发票列表。

    Args:
        ocr_results: OCR recognition results

    Returns:
        list of dict: 每张发票的信息
    """
    # 转换为内部格式
    regions = []
    for r in ocr_results:
        regions.append(OCRTextRegion(
            text=r["text"],
            confidence=r["confidence"],
            box=r["box"],
        ))

    if not regions:
        return []

    # 1. 空间聚类
    clusters = cluster_ocr_results(regions)

    # 2. 提取所有发票代码（用于后续校验）
    all_codes = extract_invoice_codes(regions)

    # 3. 逐簇解析
    all_invoices = []

    for cluster in clusters:
        inv = parse_single_cluster(cluster, all_codes)
        if inv:
            all_invoices.append(inv)

    # 4. 去重：同一面额的发票只保留一张（用户规则）
    all_invoices = deduplicate_invoices(all_invoices)

    logger.info(f"解析完成: {len(all_invoices)} 张发票")
    for inv in all_invoices:
        logger.info(f"  {inv.type.value} {inv.amount}元 号码={inv.invoice_number} "
                    f"{'✓' if not inv.corrected_by_code else '✓(代码修正)'}")

    return [inv.to_dict() for inv in all_invoices]


def parse_single_cluster(
    cluster: list[OCRTextRegion],
    all_codes: list[tuple[str, float, float]],
) -> Optional[RecognizedInvoice]:
    """
    解析一个簇（一张发票）。

    流程：
    1. 判断类型（一卡通 / 普通）
    2. 提取发票号码
    3. 提取金额
    4. 用发票代码交叉校验
    """
    full_text = " ".join(r.text for r in cluster)

    # ---- Step 1: 提取发票号码 ----
    numbers = extract_invoice_numbers(full_text)
    if not numbers:
        logger.debug(f"  簇未找到号码: {full_text[:60]}")
        return None

    # 取第一个号码
    inv_number = numbers[0]
    number_center_y = None
    number_center_x = None
    for r in cluster:
        if inv_number in r.text:
            number_center_y = r.cy
            number_center_x = r.cx
            break

    logger.debug(f"  号码: {inv_number}")

    # ---- Step 2: 判断类型和金额 ----

    # 先看是否有一卡通特征
    is_ykt, ykt_reason = is_yikatong_by_features(cluster)
    inv_type = InvoiceType.YIKATONG if is_ykt else InvoiceType.REGULAR

    # 提取金额
    amount = extract_amount_from_cluster(cluster, inv_type)

    logger.debug(f"  类型={inv_type.value} 金额={amount}元 "
                 f"(ykt_reason={ykt_reason})")

    # ---- Step 3: 发票代码交叉校验 ----
    corrected_by_code = False
    if all_codes and number_center_y:
        best_code = min(
            all_codes,
            key=lambda c: abs(c[2] - number_center_y) + abs(c[1] - number_center_x) * 0.3,
        )
        code_entry = lookup_invoice_code(best_code[0])
        if code_entry:
            if code_entry.type != inv_type or code_entry.amount != amount:
                logger.info(
                    f"  代码修正: {inv_number} "
                    f"type {inv_type.value}→{code_entry.type.value} "
                    f"amt {amount}→{code_entry.amount} "
                    f"(code={best_code[0]})"
                )
                inv_type = code_entry.type
                amount = code_entry.amount
                corrected_by_code = True

    if amount is None:
        logger.debug(f"  未找到金额: {full_text[:60]}")
        return None

    return RecognizedInvoice(
        type=inv_type,
        amount=amount,
        invoice_number=inv_number,
        corrected_by_code=corrected_by_code,
    )


def extract_amount_from_cluster(
    cluster: list[OCRTextRegion],
    inv_type: InvoiceType,
) -> Optional[int]:
    """
    从簇中提取金额。

    优先级：
    1. 发票代码（如果有）
    2. 中文大写金额
    3. 数字金额（¥X.00 格式）
    4. 一卡通根据特征猜测
    """
    full_text = " ".join(r.text for r in cluster)

    # ---- 普通发票：找 ¥X.00 格式 ----
    if inv_type == InvoiceType.REGULAR:
        # 匹配 ¥ 后的数字
        m = re.search(r'[¥￥]\s*(\d+(?:\.\d+)?)', full_text)
        if m:
            amt_str = m.group(1)
            try:
                amt = int(float(amt_str))
                if 1 <= amt <= 200:
                    return amt
            except ValueError:
                pass

    # ---- 找中文大写金额 ----
    for suffix in ["圆", "元", "拾"]:
        idx = full_text.find(suffix)
        if idx >= 0:
            # 提取包含该字符的前后文本
            start = max(0, idx - 10)
            end = min(len(full_text), idx + 5)
            context = full_text[start:end]
            amt = parse_chinese_amount(context)
            if amt and 1 <= amt <= 200:
                return amt

    # ---- 遍历所有文本找中文金额 ----
    for r in cluster:
        amt = parse_chinese_amount(r.text)
        if amt and 1 <= amt <= 200:
            return amt

    # ---- 一卡通：尝试从文本中推断 ----
    if inv_type == InvoiceType.YIKATONG:
        # 找文本中的数字金额
        nums = re.findall(r'\b(\d{1,3})\b', full_text)
        for n in nums:
            amt = int(n)
            if amt in (10, 50, 100):
                return amt

    return None


def deduplicate_invoices(invoices: list[RecognizedInvoice]) -> list[RecognizedInvoice]:
    """
    去重：同一金额的发票只保留一张。
    用户规则：每次照片中同一面额只有一张发票。
    """
    seen = set()
    result = []
    for inv in invoices:
        key = (inv.type, inv.amount)
        if key not in seen:
            seen.add(key)
            result.append(inv)
        else:
            logger.info(f"  去重: {inv.type.value} {inv.amount}元 (号码 {inv.invoice_number})")
    return result
