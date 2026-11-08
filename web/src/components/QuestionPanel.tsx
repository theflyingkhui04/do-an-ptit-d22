"use client";

import { useState } from "react";

import { ApiError, type Api, type QuestionResult } from "@/lib/api";
import { formatCell, formatMs } from "@/lib/format";

interface Props {
  api: Api;
}

const WARNINGS: Record<string, string> = {
  fabricated_number: "Câu trả lời có con số không có trong kết quả nên đã bị ẩn; xem bảng kết quả bên dưới.",
  empty_answer: "Mô hình không trả lời; xem bảng kết quả.",
  answer_failed: "Không tạo được câu trả lời; xem bảng kết quả.",
};

export default function QuestionPanel({ api }: Props) {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QuestionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.ask(question.trim()));
    } catch (exc) {
      setError(exc instanceof ApiError && exc.status === 429 ? "Đã hỏi quá nhiều, hãy thử lại sau." : (exc as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <form onSubmit={submit} className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-white p-4 sm:flex-row">
        <label htmlFor="question" className="sr-only">
          Câu hỏi về dữ liệu hoá đơn
        </label>
        <input
          id="question"
          value={question}
          maxLength={500}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ví dụ: Tổng tiền các hoá đơn tháng 8 năm 2020 là bao nhiêu?"
          className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm"
        />
        <button
          type="submit"
          disabled={busy || !question.trim()}
          className="rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {busy ? "Đang hỏi…" : "Hỏi"}
        </button>
      </form>
      {error && (
        <p role="alert" className="text-sm text-red-600">
          {error}
        </p>
      )}
      {result && (
        <section aria-label="Kết quả câu hỏi" className="space-y-3 rounded-lg border border-slate-200 bg-white p-4">
          {result.status === "error" && (
            <p role="alert" className="text-sm text-red-600">
              Không trả lời được: {result.error}
            </p>
          )}
          {result.answer && <p className="text-base text-slate-900">{result.answer}</p>}
          {result.warning && (
            <p role="status" className="text-sm text-amber-700">
              {WARNINGS[result.warning] ?? result.warning}
            </p>
          )}
          {result.sql && (
            <pre data-testid="sql" className="overflow-x-auto rounded bg-slate-900 p-3 text-xs text-slate-100">
              {result.sql}
            </pre>
          )}
          {result.columns.length > 0 && result.rows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm" aria-label="Bảng kết quả">
                <thead className="text-xs text-slate-500">
                  <tr>
                    {result.columns.map((column) => (
                      <th key={column} className="py-1 pr-3 font-medium">
                        {column}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row, index) => (
                    <tr key={index} className="border-t border-slate-100">
                      {row.map((cell, cellIndex) => (
                        <td key={cellIndex} className="py-1 pr-3">
                          {formatCell(cell)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="text-xs text-slate-500">
            {result.calls.map((call) => `${call.step}: ${call.lane} ${formatMs(call.latency_ms)}`).join(" · ")}
            {result.repairs > 0 ? " · đã sửa SQL 1 lần" : ""}
          </p>
        </section>
      )}
    </div>
  );
}
