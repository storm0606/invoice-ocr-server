"""
发票代码 → (类型, 金额) 查找表。

发票代码是印刷在发票上的固定编码，直接对应特定面额和类型。
这是从用户提供的 15 种发票代码整理而来，用于交叉校验 OCR 识别结果。
"""

from dataclasses import dataclass
from enum import Enum


class InvoiceType(str, Enum):
    YIKATONG = "一卡通"
    REGULAR = "普通发票"


@dataclass
class InvoiceCodeEntry:
    code: str
    type: InvoiceType
    amount: int


# 完整发票代码查找表
INVOICE_CODE_MAP: dict[str, InvoiceCodeEntry] = {
    # 一卡通
    "112002470103": InvoiceCodeEntry("112002470103", InvoiceType.YIKATONG, 10),
    "112002470105": InvoiceCodeEntry("112002470105", InvoiceType.YIKATONG, 50),
    "112002470106": InvoiceCodeEntry("112002470106", InvoiceType.YIKATONG, 100),
    # 普通发票
    "112002569131": InvoiceCodeEntry("112002569131", InvoiceType.REGULAR, 2),
    "112002569134": InvoiceCodeEntry("112002569134", InvoiceType.REGULAR, 3),
    "112002569135": InvoiceCodeEntry("112002569135", InvoiceType.REGULAR, 4),
    "112002569136": InvoiceCodeEntry("112002569136", InvoiceType.REGULAR, 5),
    "112002569137": InvoiceCodeEntry("112002569137", InvoiceType.REGULAR, 6),
    "112002569138": InvoiceCodeEntry("112002569138", InvoiceType.REGULAR, 7),
    "112002569139": InvoiceCodeEntry("112002569139", InvoiceType.REGULAR, 8),
    "112002569140": InvoiceCodeEntry("112002569140", InvoiceType.REGULAR, 9),
    "112002569141": InvoiceCodeEntry("112002569141", InvoiceType.REGULAR, 10),
    "112002569143": InvoiceCodeEntry("112002569143", InvoiceType.REGULAR, 20),
    "112002569144": InvoiceCodeEntry("112002569144", InvoiceType.REGULAR, 50),
    "112002569145": InvoiceCodeEntry("112002569145", InvoiceType.REGULAR, 100),
}


def lookup_invoice_code(code: str) -> InvoiceCodeEntry | None:
    """根据发票代码查询类型和金额"""
    return INVOICE_CODE_MAP.get(code)


def get_amounts_by_type(inv_type: InvoiceType) -> set[int]:
    """获取某类型发票的所有面额"""
    return {
        entry.amount
        for entry in INVOICE_CODE_MAP.values()
        if entry.type == inv_type
    }
