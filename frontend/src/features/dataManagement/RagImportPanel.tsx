import { useState } from "react";
import styles from "../../App.module.css";
import { formatCorpusType } from "../../lib/displayText";
import type { RAGDocument } from "../../types/api";

interface RagImportPanelProps {
  documents: RAGDocument[];
  busy: boolean;
  status: string | null;
  onImport: (documents: RAGDocument[]) => Promise<void>;
  onReload: () => Promise<void>;
}

export function RagImportPanel({ documents, busy, status, onImport, onReload }: RagImportPanelProps) {
  const [payload, setPayload] = useState(
    JSON.stringify(
      [
        {
          doc_id: "policy_demo_route_clearance",
          corpus: "policy",
          title: "临时通行清障规则",
          content: "当学校周边积水超过 15 厘米且接送车辆通行受限时，应优先评估步行疏散与车辆绕行方案。",
          metadata: {
            updated_at: "2026-04-02T08:00:00Z",
            tags: ["school", "transport"],
          },
        },
      ],
      null,
      2,
    ),
  );
  const [inputError, setInputError] = useState<string | null>(null);

  async function handleImport() {
    setInputError(null);

    try {
      const parsed = JSON.parse(payload) as unknown;
      const importedDocuments = Array.isArray(parsed)
        ? (parsed as RAGDocument[])
        : typeof parsed === "object" && parsed !== null && Array.isArray((parsed as { documents?: unknown }).documents)
          ? ((parsed as { documents: RAGDocument[] }).documents)
          : null;

      if (!importedDocuments) {
        throw new Error("导入 JSON 必须是数组，或包含 documents 数组字段的对象。");
      }

      await onImport(importedDocuments);
    } catch (error) {
      setInputError(error instanceof Error ? error.message : "JSON 解析失败。");
    }
  }

  return (
    <div className={styles.adminCard}>
      <div className={styles.adminCardHeader}>
        <div>
          <p className={styles.sectionLabel}>知识库导入</p>
          <h3>RAG 文档管理</h3>
        </div>
        <div className={styles.operationCounts}>
          <span>{documents.length} 份运行期文档</span>
        </div>
      </div>

      <label className={styles.fieldBlockFull}>
        <span className={styles.operationLabel}>导入内容</span>
        <textarea
          aria-label="rag-import-json"
          className={styles.codeTextarea}
          value={payload}
          onChange={(event) => setPayload(event.target.value)}
          rows={10}
          disabled={busy}
        />
      </label>

      {inputError ? <p className={styles.inlineError}>{inputError}</p> : null}
      {status ? <p className={styles.inlineStatus}>{status}</p> : null}

      <div className={styles.operationActions}>
        <button type="button" className={styles.secondaryButton} aria-label="rag-reload" disabled={busy} onClick={() => void onReload()}>
          重载运行期文档
        </button>
        <button type="button" className={styles.primaryButton} aria-label="rag-import-submit" disabled={busy} onClick={() => void handleImport()}>
          导入 JSON
        </button>
      </div>

      <div className={styles.documentList}>
        {documents.length ? (
          documents.slice(0, 6).map((document) => (
            <article key={document.doc_id} className={styles.documentCard}>
              <div className={styles.evidenceMeta}>
                <span>{formatCorpusType(document.corpus)}</span>
                <span>{document.doc_id}</span>
              </div>
              <h4>{document.title}</h4>
              <p>{document.content}</p>
            </article>
          ))
        ) : (
          <p className={styles.emptyState}>当前还没有可展示的知识库文档。</p>
        )}
      </div>
    </div>
  );
}
