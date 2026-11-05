"""Fixed Node 2 business-question catalog (Q1-Q6).

Copied verbatim from the "Business question catalog for Node 2" table in
``src/pipeline/README.md``; keep both in sync.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node2Question:
    id: str
    question: str
    preferred_views: tuple[str, ...]


QUESTIONS: tuple[Node2Question, ...] = (
    Node2Question("Q1", "Tổng chi tiêu trong tháng 08/2020 là bao nhiêu?", ("v_invoice_facts",)),
    Node2Question("Q2", "Mặt hàng nào được mua nhiều nhất?", ("v_item_facts",)),
    Node2Question("Q3", "Cửa hàng nào có tổng giá trị hóa đơn cao nhất?", ("v_invoice_facts",)),
    Node2Question("Q4", "Có bao nhiêu hóa đơn có phát sinh VAT?", ("v_invoice_facts", "v_tax_facts")),
    Node2Question("Q5", "Tổng tiền theo từng ngày lập hóa đơn là bao nhiêu?", ("v_invoice_facts",)),
    Node2Question(
        "Q6",
        "Mức thuế suất và tổng tiền thuế theo từng loại thuế là bao nhiêu?",
        ("v_tax_facts",),
    ),
)
