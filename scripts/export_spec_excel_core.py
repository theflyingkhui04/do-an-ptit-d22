"""Script to export comprehensive Software Requirements Specification and Test Matrix to Excel (.xlsx), CSV and Markdown."""
from __future__ import annotations

import csv
import io
import os
import sys
import xml.sax.saxutils as saxutils
import zipfile
from pathlib import Path


def col_name(col_idx: int) -> str:
    """Converts 0-based column index to Excel column name (A, B, ..., Z, AA, AB, ...)."""
    name = ""
    col_idx += 1
    while col_idx > 0:
        col_idx, rem = divmod(col_idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def build_sheet_xml(rows: list[list[str]], col_widths: dict[int, float] | None = None) -> str:
    """Builds OpenXML sheet XML string with optional column widths."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
    ]

    if col_widths:
        xml_parts.append('<cols>')
        for c_idx, width in col_widths.items():
            xml_parts.append(
                f'<col min="{c_idx + 1}" max="{c_idx + 1}" width="{width}" customWidth="1"/>'
            )
        xml_parts.append('</cols>')

    xml_parts.append('<sheetData>')
    for r_idx, row in enumerate(rows, 1):
        xml_parts.append(f'<row r="{r_idx}">')
        for c_idx, val in enumerate(row):
            c_ref = f"{col_name(c_idx)}{r_idx}"
            escaped = saxutils.escape(str(val) if val is not None else "")
            # Dùng inlineStr để không phụ thuộc bảng sharedStrings phức tạp
            xml_parts.append(
                f'<c r="{c_ref}" t="inlineStr"><is><t>{escaped}</t></is></c>'
            )
        xml_parts.append('</row>')
    xml_parts.append('</sheetData>')
    xml_parts.append('</worksheet>')
    return "".join(xml_parts)


def create_xlsx(sheets: dict[str, list[list[str]]], sheet_widths: dict[str, dict[int, float]], output_path: str) -> None:
    """Creates a multi-sheet XLSX workbook in pure Python."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        # [Content_Types].xml
        content_types = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
            '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '  <Default Extension="xml" ContentType="application/xml"/>',
            '  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        ]
        for idx in range(1, len(sheets) + 1):
            content_types.append(
                f'  <Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        content_types.append('</Types>')
        z.writestr("[Content_Types].xml", "".join(content_types))

        # _rels/.rels
        z.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>',
        )

        # xl/_rels/workbook.xml.rels
        wb_rels = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        ]
        for idx in range(1, len(sheets) + 1):
            wb_rels.append(
                f'  <Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
            )
        wb_rels.append('</Relationships>')
        z.writestr("xl/_rels/workbook.xml.rels", "".join(wb_rels))

        # xl/workbook.xml
        wb = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
            '  <sheets>',
        ]
        for idx, s_name in enumerate(sheets.keys(), 1):
            wb.append(f'    <sheet name="{saxutils.escape(s_name)}" sheetId="{idx}" r:id="rId{idx}"/>')
        wb.append('  </sheets>')
        wb.append('</workbook>')
        z.writestr("xl/workbook.xml", "".join(wb))

        # worksheets
        for idx, (s_name, rows) in enumerate(sheets.items(), 1):
            widths = sheet_widths.get(s_name)
            sheet_xml = build_sheet_xml(rows, col_widths=widths)
            z.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(buf.getvalue())
    print(f"✔ Đã tạo tệp Excel thành công: {output_path} ({len(buf.getvalue())} bytes)")
